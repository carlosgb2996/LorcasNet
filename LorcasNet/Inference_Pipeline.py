#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import models  # LorcasNet / LorcasNet_bn

torch.set_grad_enabled(False)

# -----------------------------------------------------------------------------
# ◉ Helpers: lectura robusta y salida del modelo
# -----------------------------------------------------------------------------
def _load_array(path: Path) -> np.ndarray:
    """Lee .csv / .csv.gz / .npy → float32."""
    p = str(path)
    try:
        if p.endswith(".npy"):
            arr = np.load(p).astype(np.float32)
        elif p.endswith(".csv.gz"):
            import gzip
            with gzip.open(p, "rt") as f:
                arr = np.loadtxt(f, delimiter=",").astype(np.float32)
        else:
            arr = np.loadtxt(p, delimiter=",").astype(np.float32)
    except Exception as e:
        raise IOError(f"No pude leer {path}: {e}")
    # Sanea NaN/Inf por seguridad
    return np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0).astype(np.float32)

def _get_flow_from_out(out):
    """
    Devuelve (flow, conf) con flow (B,2,h,w) y conf (B,1,h,w) o None.
    Acepta dicts (claves 'flow'/'conf' o similares) o tuplas/listas.
    """
    conf = None
    if isinstance(out, dict):
        flow = out.get("flow", None)
        if flow is None:
            for k in ["flows", "pred", "output"]:
                if k in out:
                    flow = out[k]; break
        conf = out.get("conf", None)
    elif isinstance(out, (list, tuple)):
        flow = out[0]
        conf = out[1] if len(out) > 1 else None
    else:
        flow = out

    if not torch.is_tensor(flow) or flow.dim() != 4 or flow.size(1) < 2:
        raise ValueError(f"Salida inválida del modelo: {type(flow)} {getattr(flow, 'shape', None)}")
    return flow, conf

# -----------------------------------------------------------------------------
# ◉ Fourier-domain phase retrieval (φ a partir de ∂φ/∂x, ∂φ/∂y)
# -----------------------------------------------------------------------------
def compute_phase(phi_x: Tensor, phi_y: Tensor, reg: float = 1e-6) -> Tensor:
    if phi_x.shape != phi_y.shape:
        raise ValueError("phi_x and phi_y must have identical shape")

    phi_x_c = torch.view_as_complex(torch.stack([phi_x, torch.zeros_like(phi_x)], dim=-1))
    phi_y_c = torch.view_as_complex(torch.stack([phi_y, torch.zeros_like(phi_y)], dim=-1))

    Fx = torch.fft.fft2(phi_x_c)
    Fy = torch.fft.fft2(phi_y_c)

    H, W = phi_x.shape[-2:]
    ky = torch.fft.fftfreq(H, d=1.0, device=phi_x.device).view(-1, 1)
    kx = torch.fft.fftfreq(W, d=1.0, device=phi_x.device).view(1, -1)
    kx2 = (2 * np.pi * kx) ** 2
    ky2 = (2 * np.pi * ky) ** 2
    denom = kx2 + ky2 + reg

    j = 1j
    numer = j * 2 * np.pi * (kx * Fx + ky * Fy)

    PHI_f = numer / denom
    phi = torch.fft.ifft2(PHI_f).real
    return phi - phi.mean(dim=(-2, -1), keepdim=True)

# -----------------------------------------------------------------------------
# ◉ Dark-field con offsets normalizados para grid_sample
# -----------------------------------------------------------------------------
def compute_dark_field(I_ref: Tensor, I_def: Tensor, flow_px_x: Tensor, flow_px_y: Tensor,
                       kernel_size: int = 15) -> Tensor:
    """
    I_ref, I_def: (B,1,H,W) en [0,1]
    flow_px_*  : (B,1,H,W) en PIXELES (no normalizados)
    Devuelve   : (B,1,H,W) mapa de dark-field (σ/μ local del residuo)
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    B, _, H, W = I_ref.shape
    device = I_ref.device

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B,H,W,2)

    # píxeles → [-1,1] para grid_sample con align_corners=True
    u_norm = (2.0 * flow_px_x.squeeze(1)) / max(W - 1, 1)
    v_norm = (2.0 * flow_px_y.squeeze(1)) / max(H - 1, 1)

    grid = base_grid.clone()
    grid[..., 0] = grid[..., 0] + u_norm
    grid[..., 1] = grid[..., 1] + v_norm

    I_warp = F.grid_sample(I_def, grid, mode="bilinear",
                           padding_mode="border", align_corners=True)

    diff = (I_ref - I_warp).abs()
    patches = diff.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    mu = patches.mean(dim=(-1, -2))
    std = patches.std(dim=(-1, -2))
    contrast = std / (mu + 1e-6)
    return contrast.unsqueeze(1)

# -----------------------------------------------------------------------------
# ◉ Métricas básicas
# -----------------------------------------------------------------------------
def _rmse(a: np.ndarray, b: np.ndarray) -> float: return float(np.sqrt(np.mean((a - b) ** 2)))
def _mae (a: np.ndarray, b: np.ndarray) -> float: return float(np.mean(np.abs(a - b)))
def _bias(a: np.ndarray, b: np.ndarray) -> float: return float(np.mean(a - b))
def _r2  (a: np.ndarray, b: np.ndarray) -> float:
    ss_res = np.sum((a - b) ** 2); ss_tot = np.sum((a - np.mean(a)) ** 2); return float(1 - ss_res / (ss_tot + 1e-12))
def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float('nan')
    va = np.var(a)
    vb = np.var(b)
    if va <= 1e-12 or vb <= 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])

def _slope_B0(g: np.ndarray, p: np.ndarray) -> float:
    """Pendiente sin intercepto que minimiza ||p - a*g||² → a = <p,g>/<g,g>."""
    g = np.asarray(g, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    den = float(np.dot(g, g))
    if den <= 1e-12:
        return float('nan')
    return float(np.dot(p, g) / den)

def compute_metrics(pred_dir: Path, gt_dir: Path, out_csv: Path | None = None, min_mag: float = 0.0):
    """
    Lee predicciones 'Def*_flow_px_{x,y}.csv' y GT 'Disp{x,y}{...}.csv'.
    Calcula métricas vectoriales y por eje, con máscara por magnitud GT > min_mag.
    """
    import csv
    pred_files = sorted(pred_dir.glob('Def*_flow_px_x.csv'))
    if not pred_files:
        warnings.warn("No prediction files found with pattern 'Def*_flow_px_x.csv'.")
        return

    rows = []
    skipped = 0

    for px in tqdm(pred_files, desc="Metrics"):
        stem = px.stem.replace('_flow_px_x', '')
        py = pred_dir / f"{stem}_flow_px_y.csv"
        if not py.exists():
            warnings.warn(f"Missing Y displacement for {stem}; skipping.")
            continue

        # GT naming: Dispx{stem.lstrip('Def')}.csv
        suffix = stem.lstrip('Def')
        gx = gt_dir / f"Dispx{suffix}.csv"
        gy = gt_dir / f"Dispy{suffix}.csv"
        if not gx.exists() or not gy.exists():
            warnings.warn(f"Missing GT for {stem}; skipping.")
            continue

        disp_px = np.loadtxt(px, delimiter=',')
        disp_py = np.loadtxt(py, delimiter=',')
        gt_x    = np.loadtxt(gx, delimiter=',')
        gt_y    = np.loadtxt(gy, delimiter=',')

        # Máscara por magnitud de GT
        if min_mag > 0.0:
            mag = np.hypot(gt_x, gt_y)
            mask = mag > float(min_mag)
        else:
            mask = np.ones_like(gt_x, dtype=bool)

        if not np.any(mask):
            skipped += 1
            continue

        # Aplanar con máscara
        px_m = disp_px[mask].ravel()
        py_m = disp_py[mask].ravel()
        gx_m = gt_x[mask].ravel()
        gy_m = gt_y[mask].ravel()

        # Vectoriales (concat X/Y)
        p_vec = np.concatenate([px_m, py_m])
        g_vec = np.concatenate([gx_m, gy_m])

        rmse = _rmse(g_vec, p_vec)
        mae  = _mae (g_vec, p_vec)
        bias = _bias(p_vec, g_vec)

        # R² seguro
        ss_res = np.sum((g_vec - p_vec) ** 2)
        ss_tot = np.sum((g_vec - np.mean(g_vec)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-12))

        # Por eje
        rmse_x = _rmse(gx_m, px_m); rmse_y = _rmse(gy_m, py_m)
        mae_x  = _mae (gx_m, px_m); mae_y  = _mae (gy_m, py_m)
        bias_x = _bias(px_m, gx_m); bias_y = _bias(py_m, gy_m)

        # Correlaciones y pendientes B0
        r_x = _safe_corr(px_m, gx_m)
        r_y = _safe_corr(py_m, gy_m)
        r_mag = _safe_corr(np.hypot(px_m, py_m), np.hypot(gx_m, gy_m))

        ax = _slope_B0(gx_m, px_m)
        ay = _slope_B0(gy_m, py_m)

        rows.append((
            stem, rmse, mae, bias, r2,
            rmse_x, mae_x, bias_x,
            rmse_y, mae_y, bias_y,
            r_x, r_y, r_mag,
            ax, ay, px_m.size  # n_mask
        ))

    if not rows:
        warnings.warn(f"No metrics computed (empty rows). Skipped={skipped}.")
        return

    header = [
        'Identifier',
        'RMSE','MAE','Bias','R2',
        'RMSE_x','MAE_x','Bias_x',
        'RMSE_y','MAE_y','Bias_y',
        'Pearson_x','Pearson_y','Pearson_mag',
        'SlopeB0_x','SlopeB0_y',
        'N_mask'
    ]

    rows_np = np.asarray(rows, dtype=object)
    means = np.mean(np.vstack([rows_np[:,1+i].astype(float) for i in range(len(header)-2)]).T, axis=0)

    if out_csv is not None:
        with out_csv.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"✅ Metrics written to {out_csv}  (skipped={skipped})")
    # Resumen legible
    print("\n=== Metrics summary (masked) ===")
    print(f" count={len(rows)}  skipped_empty_mask={skipped}  min_mag={min_mag:.4f}")
    print(f" RMSE={means[0]:.4f}  MAE={means[1]:.4f}  Bias={means[2]:.5f}  R²={means[3]:.4f}")
    print(f"  · X: RMSE={means[4]:.4f} MAE={means[5]:.4f} Bias={means[6]:.5f}  r={means[10]:.3f}  aB0={means[13]:.3f}")
    print(f"  · Y: RMSE={means[7]:.4f} MAE={means[8]:.4f} Bias={means[9]:.5f}  r={means[11]:.3f}  aB0={means[14]:.3f}")
    print(f"  · |v|: r={means[12]:.3f}")


# -----------------------------------------------------------------------------
# ◉ Calibración opcional en inference
# -----------------------------------------------------------------------------
def apply_calibration(flow_x: Tensor, flow_y: Tensor,
                      mode: str = "none",
                      gain: float | None = None,
                      ax: float | None = None, ay: float | None = None,
                      bx: float = 0.0, by: float = 0.0) -> tuple[Tensor, Tensor]:
    """
    flow_* en píxeles. Devuelve flows calibrados.
    mode: 'none' | 'A' | 'B' | 'B0'
    """
    mode = (mode or "none").upper()
    if mode == "NONE":
        return flow_x, flow_y
    if mode == "A":
        if gain is None:
            raise ValueError("Calibración A requiere --gain")
        return gain * flow_x, gain * flow_y
    if mode in ("B", "B0"):
        if ax is None or ay is None:
            raise ValueError("Calibración B/B0 requiere --ax y --ay")
        bx_ = 0.0 if mode == "B0" else float(bx)
        by_ = 0.0 if mode == "B0" else float(by)
        return ax * flow_x + bx_, ay * flow_y + by_
    raise ValueError(f"Modo de calibración desconocido: {mode}")

# -----------------------------------------------------------------------------
# ◉ Inference
# -----------------------------------------------------------------------------
def run_inference(data_dir: Path, save_dir: Path, model_ckpt: Path,
                  compute_phase_dark: bool = True,
                  cal_mode: str = "none",
                  gain: float | None = None,
                  ax: float | None = None, ay: float | None = None,
                  bx: float = 0.0, by: float = 0.0,
                  save_conf_per_pair: bool = False,
                  # ▼ nuevos flags ya soportados en tu main
                  phase_scale_x: float = 1.0,
                  phase_scale_y: float = 1.0,
                  wavelength: float = 0.025,
                  dark_ksize: int = 15,
                  save_stats: bool = False,
                  # ▼ NUEVO: neutralizar la capa de calibración final
                  neutralize_calib: bool = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Modelo (flow en PÍXELES con la cabeza lineal)
    net = models.LorcasNet_bn()
    ckpt = torch.load(model_ckpt, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    missing = net.load_state_dict(state, strict=False)
    if getattr(missing, "missing_keys", None) or getattr(missing, "unexpected_keys", None):
        print("⚠️ load_state_dict mismatches:", missing)
        # ❌ aborta si faltan pesos de las flow-heads
        if getattr(missing, "missing_keys", None) and any("predict_flow" in k for k in missing.missing_keys):
            raise RuntimeError(f"❌ Faltan pesos de flow-heads en el checkpoint: {missing.missing_keys}")

    # ✅ neutraliza la calibración de salida si se pide
    if neutralize_calib and hasattr(net, 'out_scale') and hasattr(net, 'out_bias'):
        with torch.no_grad():
            try:
                net.out_scale.fill_(1.0)
                net.out_bias.fill_(0.0)
                print("ℹ️  Neutralizada calibración de salida (out_scale=1, out_bias=0)")
            except Exception as e:
                print("⚠️  No pude neutralizar out_scale/out_bias:", e)

    net.to(device).eval()

    if getattr(missing, "missing_keys", None) or getattr(missing, "unexpected_keys", None):
        print("⚠️ load_state_dict mismatches:", missing)
    net.to(device).eval()

    # Pares Ref/Def
    pairs = [(p, data_dir / p.name.replace('Ref', 'Def'))
             for p in list(data_dir.glob('*Ref*.csv')) + list(data_dir.glob('*Ref*.csv.gz')) + list(data_dir.glob('*Ref*.npy'))
             if (data_dir / p.name.replace('Ref', 'Def')).exists()]
    if not pairs:
        raise FileNotFoundError("No Ref/Def pairs found in data_dir (busqué .csv/.csv.gz/.npy).")

    save_dir.mkdir(parents=True, exist_ok=True)
    stats_rows = []

    k = 2 * np.pi / float(wavelength)

    autocast_kwargs = dict(enabled=device.type == 'cuda', dtype=torch.float16 if device.type == 'cuda' else None)
    with torch.inference_mode():
        for ref_path, def_path in tqdm(pairs, desc="Inference"):
            try:
                ref_np = _load_array(ref_path) / 255.0
                def_np = _load_array(def_path) / 255.0
            except Exception as e:
                warnings.warn(f"Could not read {ref_path} / {def_path}: {e}"); continue

            H, W = ref_np.shape
            inp = torch.from_numpy(np.stack([ref_np, def_np])).unsqueeze(0).to(device)  # [1,2,H,W]

            with torch.autocast(device_type=device.type, **{k: v for k, v in autocast_kwargs.items() if v is not None}):
                out = net(inp)
                flow2, conf2 = _get_flow_from_out(out)  # flow2 en píxeles

            # Asegurar tamaño H×W solo si hace falta
            if flow2.shape[-2:] != (H, W):
                flow2 = F.interpolate(flow2, size=(H, W), mode='bilinear', align_corners=False)
            if conf2 is not None and conf2.shape[-2:] != (H, W):
                conf2 = F.interpolate(conf2, size=(H, W), mode='bilinear', align_corners=False)

            # float32 estable para post-procesado/guardado
            flow2 = flow2.float()
            if conf2 is not None:
                conf2 = torch.clamp(conf2.float(), min=1e-6, max=1e3)

            flow_px_x = flow2[0, 0].clone()
            flow_px_y = flow2[0, 1].clone()

            # Calibración opcional (A/B/B0)
            flow_px_x, flow_px_y = apply_calibration(
                flow_px_x, flow_px_y,
                mode=cal_mode, gain=gain, ax=ax, ay=ay, bx=bx, by=by
            )

            stem = def_path.stem  # compatible con .csv/.csv.gz/.npy

            np.savetxt(save_dir / f"{stem}_flow_px_x.csv", flow_px_x.cpu().numpy(), delimiter=',')
            np.savetxt(save_dir / f"{stem}_flow_px_y.csv", flow_px_y.cpu().numpy(), delimiter=',')

            # Guardar confianza (por-par) si se solicita
            if save_conf_per_pair and conf2 is not None:
                np.savetxt(save_dir / f"{stem}_conf.csv", conf2[0, 0].detach().cpu().numpy(), delimiter=',')

            # Último mapa por comodidad
            run_inference._last_conf_map = None if conf2 is None else conf2[0, 0].detach().cpu().numpy()

            # Stats opcionales (rápidos) para inspección de “estructura”
            if save_stats:
                fx = flow_px_x.detach().cpu().numpy()
                fy = flow_px_y.detach().cpu().numpy()
                row = {
                    "id": stem,
                    "fx_mean": float(fx.mean()), "fx_std": float(fx.std()), "fx_maxabs": float(np.max(np.abs(fx))),
                    "fy_mean": float(fy.mean()), "fy_std": float(fy.std()), "fy_maxabs": float(np.max(np.abs(fy))),
                }
                if conf2 is not None:
                    c = conf2[0,0].detach().cpu().numpy()
                    row.update({
                        "conf_mean": float(np.mean(c)),
                        "conf_p50": float(np.percentile(c, 50)),
                        "conf_p95": float(np.percentile(c, 95)),
                    })
                stats_rows.append(row)

            if not compute_phase_dark:
                continue

            # Phase retrieval: si el flujo representa proporcionalmente ∂φ/∂x,∂φ/∂y
            # puedes ajustar phase_scale_x/y y/o wavelength.
            phi = compute_phase(
                (k * phase_scale_x * flow_px_x).to(device),
                (k * phase_scale_y * flow_px_y).to(device),
                reg=1e-6
            )
            np.savetxt(save_dir / f"{stem}_phase.csv", phi.cpu().numpy(), delimiter=',')

            # Dark-field
            ref_t = torch.from_numpy(ref_np).unsqueeze(0).unsqueeze(0).to(device)
            def_t = torch.from_numpy(def_np).unsqueeze(0).unsqueeze(0).to(device)
            u_px  = flow_px_x.unsqueeze(0).unsqueeze(0)
            v_px  = flow_px_y.unsqueeze(0).unsqueeze(0)
            Cmap = compute_dark_field(ref_t, def_t, u_px, v_px, kernel_size=dark_ksize)
            np.savetxt(save_dir / f"{stem}_darkfield.csv", Cmap.squeeze().cpu().numpy(), delimiter=',')

    # Guardar resumen de stats si se pidió
    if save_stats and stats_rows:
        import csv
        out_csv = save_dir / "summary_stats.csv"
        keys = list(stats_rows[0].keys())
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(stats_rows)
        print(f"📝 Stats por-par guardados en {out_csv}")

    print("✅ Inference finished. Results in", save_dir)

# -----------------------------------------------------------------------------
# ◉ CLI
# -----------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="LorcasNet inference / metrics pipeline")
    sub = p.add_subparsers(dest="mode", required=True)

    # Inference --------------------------------------------------------------
    p_inf = sub.add_parser('infer', help="Run network inference")
    p_inf.add_argument('--data_dir', required=True, type=Path)
    p_inf.add_argument('--save_dir', required=True, type=Path)
    p_inf.add_argument('--ckpt',     required=True, type=Path)
    p_inf.add_argument('--no_phase_dark', action='store_true')
    p_inf.add_argument('--neutralize-calib', action='store_true')
    # Calibración opcional
    p_inf.add_argument('--cal-mode', choices=['none','A','B','B0'], default='none')
    p_inf.add_argument('--gain',  type=float, default=None, help='Ganancia global (A)')
    p_inf.add_argument('--ax',    type=float, default=None, help='Pendiente X (B/B0)')
    p_inf.add_argument('--ay',    type=float, default=None, help='Pendiente Y (B/B0)')
    p_inf.add_argument('--bx',    type=float, default=0.0,  help='Intercepto X (B)')
    p_inf.add_argument('--by',    type=float, default=0.0,  help='Intercepto Y (B)')
    p_inf.add_argument('--save-conf', action='store_true', help='Guardar mapa(s) de confianza por par')
    # Ajustes físicos / DF / stats
    p_inf.add_argument('--phase-scale-x', type=float, default=1.0, help='Escala multiplicativa sobre flow_x para phase retrieval')
    p_inf.add_argument('--phase-scale-y', type=float, default=1.0, help='Escala multiplicativa sobre flow_y para phase retrieval')
    p_inf.add_argument('--wavelength',    type=float, default=0.025, help='Longitud de onda (mm o unidades consistentes con el modelo)')
    p_inf.add_argument('--dark-ksize',    type=int,   default=15, help='Kernel del dark-field (impar)')
    p_inf.add_argument('--save-stats',    action='store_true', help='Guardar summary_stats.csv con estadísticas por par')

    # Metrics ---------------------------------------------------------------
    p_met = sub.add_parser('metrics', help="Compute metrics on displacement CSVs")
    p_met.add_argument('--pred_dir', required=True, type=Path)
    p_met.add_argument('--gt_dir',   required=True, type=Path)
    p_met.add_argument('--out_csv',  type=Path, required=True)

    # Both ------------------------------------------------------------------
    p_both = sub.add_parser('both', help="Run inference then metrics")
    p_both.add_argument('--data_dir', required=True, type=Path)
    p_both.add_argument('--save_dir', required=True, type=Path)
    p_both.add_argument('--ckpt',     required=True, type=Path)
    p_both.add_argument('--gt_dir',   required=True, type=Path)
    p_both.add_argument('--no_phase_dark', action='store_true')
    p_both.add_argument('--cal-mode', choices=['none','A','B','B0'], default='none')
    p_both.add_argument('--gain',  type=float, default=None)
    p_both.add_argument('--ax',    type=float, default=None)
    p_both.add_argument('--ay',    type=float, default=None)
    p_both.add_argument('--bx',    type=float, default=0.0)
    p_both.add_argument('--by',    type=float, default=0.0)
    p_both.add_argument('--save-conf', action='store_true')
    p_both.add_argument('--phase-scale-x', type=float, default=1.0)
    p_both.add_argument('--phase-scale-y', type=float, default=1.0)
    p_both.add_argument('--wavelength',    type=float, default=0.025)
    p_both.add_argument('--dark-ksize',    type=int,   default=15)
    p_both.add_argument('--save-stats',    action='store_true')
    p_both.add_argument('--out_csv',  type=Path, required=True)
    p_both.add_argument('--neutralize-calib', action='store_true')
    p_met.add_argument('--min_mag', type=float, default=0.0,
                   help='Umbral de magnitud GT (px) para enmascarar píxeles débiles (p.ej., 0.05)')

    p_both.add_argument('--min_mag', type=float, default=0.0,
                    help='Umbral de magnitud GT (px) para enmascarar píxeles débiles (p.ej., 0.05)')

    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    if args.mode == 'infer':
        run_inference(args.data_dir, args.save_dir, args.ckpt,
                      compute_phase_dark=not args.no_phase_dark,
                      cal_mode=args.cal_mode, gain=args.gain,
                      ax=args.ax, ay=args.ay, bx=args.bx, by=args.by,
                      save_conf_per_pair=getattr(args, 'save_conf', False),
                      phase_scale_x=getattr(args, 'phase_scale_x', 1.0),
                      phase_scale_y=getattr(args, 'phase_scale_y', 1.0),
                      wavelength=getattr(args, 'wavelength', 0.025),
                      dark_ksize=getattr(args, 'dark_ksize', 15),
                      save_stats=getattr(args, 'save_stats', False),
                      neutralize_calib=getattr(args, 'neutralize_calib', False))

        if getattr(args, 'save_conf', False):
            conf = getattr(run_inference, "_last_conf_map", None)
            if conf is not None:
                np.savetxt(Path(args.save_dir) / "last_confidence_map.csv", conf, delimiter=',')

    elif args.mode == 'metrics':
        compute_metrics(args.pred_dir, args.gt_dir, args.out_csv, min_mag=args.min_mag)

    elif args.mode == 'both':
        run_inference(args.data_dir, args.save_dir, args.ckpt,
                      compute_phase_dark=not args.no_phase_dark,
                      cal_mode=args.cal_mode, gain=args.gain,
                      ax=args.ax, ay=args.ay, bx=args.bx, by=args.by,
                      save_conf_per_pair=getattr(args, 'save_conf', False),
                      phase_scale_x=getattr(args, 'phase_scale_x', 1.0),
                      phase_scale_y=getattr(args, 'phase_scale_y', 1.0),
                      wavelength=getattr(args, 'wavelength', 0.025),
                      dark_ksize=getattr(args, 'dark_ksize', 15),
                      save_stats=getattr(args, 'save_stats', False),
                      neutralize_calib=getattr(args, 'neutralize_calib', False))

        if getattr(args, 'save_conf', False):
            conf = getattr(run_inference, "_last_conf_map", None)
            if conf is not None:
                np.savetxt(Path(args.save_dir) / "last_confidence_map.csv", conf, delimiter=',')

        compute_metrics(args.save_dir, args.gt_dir, args.out_csv, min_mag=args.min_mag)
