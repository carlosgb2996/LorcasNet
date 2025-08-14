#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generador de conjuntos de prueba para redes que estiman campos de desplazamiento
sobre patrones de moteado (speckle).

• Deformaciones suaves gaussianas (warp denso)
• Ruido Poisson opcional (rango 0–255)
• CSVs: Ref_, Def_, Dispx_, Dispy_
• Reproducible por semilla/worker/augment
"""

import os
import re
import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
from numpy.random import default_rng, Generator
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage import io, img_as_float32
from skimage.color import rgb2gray
from tqdm import tqdm
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

def _extract_index(path: Path) -> int:
    """Extrae el último bloque de dígitos del nombre (Ref000.tif → 000)."""
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else 0


def generate_smooth_displacement_field(rg: Generator, subset_size: int,
                                       sigma: float, amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    """Ruido gaussiano → filtro gaussiano (reflect). Devuelve float32."""
    nx = rg.normal(0.0, amplitude, (subset_size, subset_size)).astype(np.float32, copy=False)
    ny = rg.normal(0.0, amplitude, (subset_size, subset_size)).astype(np.float32, copy=False)
    dx = gaussian_filter(nx, sigma=sigma, mode="reflect").astype(np.float32, copy=False)
    dy = gaussian_filter(ny, sigma=sigma, mode="reflect").astype(np.float32, copy=False)
    return dx, dy


def cap_max_abs(dx: np.ndarray, dy: np.ndarray, max_disp: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:

    """Reescala (dx,dy) para que max(|dx|,|dy|) ≤ max_disp (si max_disp>0)."""
    if not max_disp or max_disp <= 0:
        return dx, dy
    m = max(float(np.max(np.abs(dx))), float(np.max(np.abs(dy))), 1e-12)
    s = max_disp / m
    return (dx * s).astype(np.float32, copy=False), (dy * s).astype(np.float32, copy=False)


def apply_poisson_noise(rg: Generator, img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Añade ruido Poisson en rango 0–255. Si scale<=0, retorna copia clipeada.
    """
    img = img.astype(np.float32, copy=False)
    if scale <= 0:
        return np.clip(img, 0, 255, out=img.copy())
    lo = float(np.nanmin(img)); hi = float(np.nanmax(img))
    img_pos = np.clip(img, 0, None, out=img.copy())
    noisy = rg.poisson(img_pos * scale).astype(np.float32, copy=False) / float(scale)
    return np.clip(noisy, lo, hi, out=noisy)


def warp_dense(ref_noisy: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """
    Warpeo denso bicúbico (order=3) con padding reflect y margen de 2 px.
    """
    H, W = ref_noisy.shape
    ref_pad = np.pad(ref_noisy, 2, mode="reflect")
    Y, X = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    coords = np.stack([Y + dy + 2.0, X + dx + 2.0], axis=0).astype(np.float32, copy=False)
    return map_coordinates(ref_pad, coords, order=3, mode="reflect").astype(np.float32, copy=False)


def save_outputs(out_dir: Path, idx: int, i: int, ref_noisy, bd, dx, dy) -> None:
    prefix = f"{idx:03d}_{i:03d}"
    np.savetxt(out_dir / f"Ref_{prefix}.csv",   ref_noisy.astype(np.float32), delimiter=",", fmt="%.6f")
    np.savetxt(out_dir / f"Def_{prefix}.csv",   bd.astype(np.float32),        delimiter=",", fmt="%.6f")
    np.savetxt(out_dir / f"Dispx_{prefix}.csv", dx.astype(np.float32),        delimiter=",", fmt="%.6f")
    np.savetxt(out_dir / f"Dispy_{prefix}.csv", dy.astype(np.float32),        delimiter=",", fmt="%.6f")


# ──────────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────────

def process_ref(args):
    path, out_dir, cfg = args
    path = Path(path)

    # Semillas reproducibles:
    base_seed = 0 if cfg.seed is None else int(cfg.seed)
    idx_file = _extract_index(path)
    pid = os.getpid()
    rg_file = default_rng((base_seed ^ (idx_file << 12) ^ pid) & 0xFFFFFFFF)

    # Cargar imagen y convertir a 0–255 float32
    img = io.imread(path)
    if img.ndim == 3:
        img = rgb2gray(img)
    img = img_as_float32(img)  # 0–1
    ref = (img * 255.0).astype(np.float32, copy=False)

    if ref.shape != (cfg.subset_size, cfg.subset_size):
        raise ValueError(f"{path.name}: shape {ref.shape} ≠ subset_size={cfg.subset_size}")

    # Ruido sobre la referencia
    ref_noisy = apply_poisson_noise(rg_file, ref, cfg.noise_scale)
    if float(ref_noisy.std()) < 1e-6:
        raise ValueError(f"{path.name}: referencia plana tras Poisson (noise_scale={cfg.noise_scale}).")

    logs = []
    for i in range(1, cfg.n_aug + 1):
        rg_aug = default_rng((base_seed ^ (idx_file << 16) ^ (i * 0x9E3779B1) ^ pid) & 0xFFFFFFFF)

        dx, dy = generate_smooth_displacement_field(rg_aug, cfg.subset_size, cfg.sigma, cfg.amplitude)
        dx, dy = cap_max_abs(dx, dy, cfg.max_disp)

        bd = warp_dense(ref_noisy, dx, dy)
        bd = apply_poisson_noise(rg_aug, bd, cfg.noise_scale)

        if float(bd.std()) < 1e-6:
            raise ValueError(f"{path.name}: deformada plana en aug={i}.")

        save_outputs(out_dir, idx_file, i, ref_noisy, bd, dx, dy)

        if (cfg.log_every > 0) and (i == 1 or (i % cfg.log_every == 0)):
            logs.append(
                f"[{path.name} | aug {i:02d}] "
                f"Ref μ={ref_noisy.mean():.3f} σ={ref_noisy.std():.3f} | "
                f"Def μ={bd.mean():.3f} σ={bd.std():.3f} | "
                f"|dx|_max={np.max(np.abs(dx)):.3f} |dy|_max={np.max(np.abs(dy)):.3f}"
            )
    return logs


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Augmentación de patrones de moteado (dataset de métricas)")
    ap.add_argument("--refs_dir",    type=Path, required=True, help="Carpeta con Ref*.tif/.tiff/.png")
    ap.add_argument("--out_dir",     type=Path, required=True, help="Directorio de salida")
    ap.add_argument("--n_aug",       type=int,   default=15,   help="Variaciones por referencia")
    ap.add_argument("--subset_size", type=int,   default=256,  help="Tamaño H=W")
    ap.add_argument("--sigma",       type=float, default=7.5,  help="Suavizado gaussiano (px)")
    ap.add_argument("--amplitude",   type=float, default=0.6,  help="STD del ruido inicial")
    ap.add_argument("--max_disp",    type=float, default=0.6,  help="Máximo |disp| tras suavizar (0=sin límite)")
    ap.add_argument("--noise_scale", type=float, default=1.0,  help="Escala para Poisson (rango 0–255)")
    ap.add_argument("--workers",     type=int,   default=cpu_count(), help="Procesos en paralelo")
    ap.add_argument("--seed",        type=int,   default=None, help="Semilla base reproducible")
    ap.add_argument("--log_every",   type=int,   default=0,    help="Imprime stats cada N (0 = silenciar)")
    ap.add_argument("--quiet",       action="store_true",      help="Silenciar logs por imagen")
    cfg = ap.parse_args()

    # Salida limpia
    if cfg.out_dir.exists():
        shutil.rmtree(cfg.out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    refs = sorted(
        list(cfg.refs_dir.glob("Ref*.tif")) +
        list(cfg.refs_dir.glob("Ref*.tiff")) +
        list(cfg.refs_dir.glob("Ref*.png"))
    )
    if not refs:
        raise FileNotFoundError(f"No se encontraron referencias en {cfg.refs_dir}")

    tasks = [(str(r), cfg.out_dir, cfg) for r in refs]

    with Pool(cfg.workers) as pool, tqdm(total=len(tasks), desc="Procesando", unit="img") as bar:
        for logs in pool.imap_unordered(process_ref, tasks):
            bar.update()
            if logs and not cfg.quiet:
                for line in logs:
                    bar.write(line)

    total = len(refs) * cfg.n_aug
    print(f"✅ Generadas {total} deformaciones y mapas de desplazamiento.")

    # Anotaciones Ref-Def para facilitar pairing en pipelines
    ann_path = cfg.out_dir / "Metric_annotations.csv"
    with open(ann_path, "w") as f:
        for ref_file in sorted(cfg.out_dir.glob("Ref_*.csv")):
            def_file = ref_file.name.replace("Ref_", "Def_")
            f.write(f"{ref_file.name},{def_file}\n")
    print(f"✅ Anotaciones guardadas en {ann_path.name}")


if __name__ == "__main__":
    main()
