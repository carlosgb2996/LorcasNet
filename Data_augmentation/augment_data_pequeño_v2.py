#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data augmentation para patrones speckle (v3)
- Modos de deformación: global / local (1 parche) / mix (global + K parches)
- Ruido Poisson + (opcional) ruido Gaussiano de lectura
- (Opcional) máscara de atenuación suave
- Optimizado para HPC: menos overhead de multiprocessing y menos I/O
"""
import os
import argparse
import shutil
import math
from pathlib import Path
from functools import lru_cache
from multiprocessing import Pool, cpu_count
import gzip

import numpy as np
from numpy.random import default_rng
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage import io
from skimage.color import rgb2gray

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades numéricas
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def raised_cosine_2d(n: int) -> np.ndarray:
    """Ventana Hann 2D normalizada en [0,1], tamaño n x n."""
    if n <= 1:
        return np.ones((n, n), dtype=np.float32)
    i = np.arange(n, dtype=np.float32)
    w1d = 0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n - 1))
    w2d = np.outer(w1d, w1d).astype(np.float32)
    w2d /= (w2d.max() + 1e-12)
    return w2d


def add_poisson_noise(img: np.ndarray, scale: float = 1.0, rng=None) -> np.ndarray:
    """Ruido Poisson preservando rango [0,255] (float32)."""
    if rng is None:
        rng = default_rng()
    img_s = np.clip(img, 0, None) * float(scale)
    noisy = rng.poisson(img_s.astype(np.float64)).astype(np.float32)  # poisson necesita float64
    noisy = noisy / max(scale, 1e-12)
    return np.clip(noisy, 0, 255).astype(np.float32)


def add_read_noise(img: np.ndarray, sigma: float, rng=None) -> np.ndarray:
    """Ruido Gaussiano de lectura (additivo), sigma en niveles 0..255."""
    if sigma <= 0:
        return img
    if rng is None:
        rng = default_rng()
    noisy = img + rng.normal(0.0, float(sigma), img.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.float32)


def apply_deformation(ref_pad: np.ndarray, X: np.ndarray, Y: np.ndarray,
                      dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Warp de imagen completa con padding de 2 px (order=3)."""
    coords = np.stack([Y + dy + 2, X + dx + 2])
    # map_coordinates devuleve float64; convertimos a float32
    out = map_coordinates(ref_pad, coords, order=3, mode='reflect')
    return out.astype(np.float32)


def affine_field(X: np.ndarray, Y: np.ndarray, tx, ty, a, b, c, d) -> tuple[np.ndarray, np.ndarray]:
    """
    Desplazamiento afín pequeño:
      [dx] = [a b][X] + [tx]
      [dy]   [c d][Y]   [ty]
    a,b,c,d en px/px, tx,ty en px (valores pequeños).
    """
    dx = a * X + b * Y + tx
    dy = c * X + d * Y + ty
    return dx.astype(np.float32), dy.astype(np.float32)


def smooth_unit_field_like(shape, sigma, rng):
    """Campo suave en [-1,1] aprox, mediante blur y normalización por máximo absoluto."""
    f = rng.normal(0.0, 1.0, shape).astype(np.float32)
    f = gaussian_filter(f, sigma, mode="reflect").astype(np.float32)
    m = float(np.max(np.abs(f))) + 1e-8
    return (f / m).astype(np.float32)


def attenuation_mask(H, W, amp, rng, sigma=32.0):
    """
    Máscara multiplicativa suave ≈ 1 + amp * u(x,y), con u ~ [-1,1] suave.
    amp típico: 0.05 (±5%). Clampeado a ≥0.
    """
    if amp <= 0:
        return np.ones((H, W), dtype=np.float32)
    u = smooth_unit_field_like((H, W), sigma=sigma, rng=rng)  # [-1,1]
    m = 1.0 + float(amp) * u
    return np.clip(m, 0.0, None).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Campos de desplazamiento
# ──────────────────────────────────────────────────────────────────────────────

def generate_single_patch_displacement(patch_sizes, subset_size,
                                       max_disp=0.6, rng=None,
                                       blend_sigma=0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Un parche local suave (bordes con ventana coseno 2D), |dx|,|dy| ≤ max_disp.
    blend_sigma>0 aplica un suavizado global suave al campo para “desdibujar” la caja.
    """
    assert rng is not None
    patch_size = int(rng.choice(patch_sizes))
    x0 = int(rng.integers(0, subset_size - patch_size + 1))
    y0 = int(rng.integers(0, subset_size - patch_size + 1))

    fx = rng.normal(0.0, 1.0, (patch_size, patch_size)).astype(np.float32)
    fy = rng.normal(0.0, 1.0, (patch_size, patch_size)).astype(np.float32)
    s = max(1.0, patch_size / 6.0)  # kernel proporcional
    fx = gaussian_filter(fx, s, mode="reflect").astype(np.float32)
    fy = gaussian_filter(fy, s, mode="reflect").astype(np.float32)

    eps = 1e-8
    fx = fx / (np.max(np.abs(fx)) + eps)
    fy = fy / (np.max(np.abs(fy)) + eps)

    w = raised_cosine_2d(patch_size)  # 0..1
    fx *= w
    fy *= w

    amp = float(rng.uniform(0.3, 1.0)) * float(max_disp)
    fx *= amp
    fy *= amp

    disp_x = np.zeros((subset_size, subset_size), dtype=np.float32)
    disp_y = np.zeros((subset_size, subset_size), dtype=np.float32)
    disp_x[y0:y0 + patch_size, x0:x0 + patch_size] = fx
    disp_y[y0:y0 + patch_size, x0:x0 + patch_size] = fy

    if blend_sigma and blend_sigma > 0:
        disp_x = gaussian_filter(disp_x, blend_sigma, mode="reflect").astype(np.float32)
        disp_y = gaussian_filter(disp_y, blend_sigma, mode="reflect").astype(np.float32)
    return disp_x, disp_y


def generate_multi_patch_displacement(patch_sizes, subset_size, k_min, k_max,
                                      max_disp=0.6, rng=None, blend_sigma=1.0):
    """Suma de K parches locales (K∈[k_min,k_max])."""
    K = int(rng.integers(k_min, k_max + 1))
    dx = np.zeros((subset_size, subset_size), dtype=np.float32)
    dy = np.zeros((subset_size, subset_size), dtype=np.float32)
    for _ in range(K):
        px, py = generate_single_patch_displacement(
            patch_sizes, subset_size, max_disp=max_disp, rng=rng, blend_sigma=0.0
        )
        dx += px; dy += py
    # suavizado de mezcla para borrar cajas residuales
    if blend_sigma and blend_sigma > 0:
        dx = gaussian_filter(dx, blend_sigma, mode="reflect").astype(np.float32)
        dy = gaussian_filter(dy, blend_sigma, mode="reflect").astype(np.float32)
    return dx, dy


def generate_global_displacement(subset_size, sigma, amplitude, max_disp, rng=None,
                                 affine_prob=0.5, affine_mag=0.003):
    """
    Campo global suave + (opcional) componente afín pequeño.
    - sigma (px) controla suavidad
    - amplitude (px) es cota nominal (Usa U(0.3,1.0)*amplitude)
    - max_disp (px) recorta extremos
    - affine_prob: prob de sumar un término afín (a,b,c,d ~ U(-m,m), tx,ty ~ U(-m*H, m*W))
    """
    assert rng is not None
    H = W = subset_size
    u = smooth_unit_field_like((H, W), sigma=sigma, rng=rng)
    v = smooth_unit_field_like((H, W), sigma=sigma, rng=rng)

    amp = float(rng.uniform(0.3, 1.0)) * float(amplitude)
    dx = (amp * u).astype(np.float32)
    dy = (amp * v).astype(np.float32)

    # Afín pequeño (traslación/rot/cizalla suaves)
    if rng.uniform() < affine_prob:
        X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        m = float(affine_mag)
        a, b, c, d = rng.uniform(-m, m, 4)  # px/px
        tx = rng.uniform(-m * W, m * W)     # px
        ty = rng.uniform(-m * H, m * H)     # px
        dax, day = affine_field(X, Y, tx, ty, a, b, c, d)
        dx += dax.astype(np.float32); dy += day.astype(np.float32)

    dx = np.clip(dx, -max_disp, max_disp).astype(np.float32)
    dy = np.clip(dy, -max_disp, max_disp).astype(np.float32)
    return dx, dy


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_array(path: Path, arr: np.ndarray, fmt: str, compress: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(str(path.with_suffix(".npy")), arr.astype(np.float32))
    else:
        if compress:
            with gzip.open(str(path) + ".gz", "wt") as f:
                np.savetxt(f, arr.astype(np.float32), delimiter=",", fmt="%.6f")
        else:
            np.savetxt(str(path), arr.astype(np.float32), delimiter=",", fmt="%.6f")


def save_data(out_dir: Path, idx: int, l: int, ref_noisy, bd_noisy, dx, dy,
              fmt: str = "csv", compress: bool = False):
    names = [
        f"Ref{idx:03d}_{l:03d}.csv",
        f"Def{idx:03d}_{l:03d}.csv",
        f"Dispx{idx:03d}_{l:03d}.csv",
        f"Dispy{idx:03d}_{l:03d}.csv"
    ]
    # Guardar
    save_array(out_dir / names[0], ref_noisy, fmt, compress)
    save_array(out_dir / names[1], bd_noisy,  fmt, compress)
    save_array(out_dir / names[2], dx,        fmt, compress)
    save_array(out_dir / names[3], dy,        fmt, compress)
    return names


# ──────────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────────

def process_image(args):
    (path, train_dir, test_dir, n_train, n_test,
     subset_size, patch_sizes, ann_tmp_dir,
     p_global, p_local, p_mix, global_sigma, global_amp, noise_scale,
     max_disp_local, max_disp_global, seed_base, k_ref, n_refs_total,
     mode_blend_sigma, mix_kmin, mix_kmax,
     gaussian_read_std, attenuation_amp,
     save_fmt, compress) = args

    # id único por referencia para reproducibilidad
    try:
        idx = int(Path(path).stem.split('_')[-1])
    except Exception:
        idx = int(100000 + k_ref)
    rng = default_rng(seed_base + idx)

    img = io.imread(path)
    if img.ndim == 3:
        img = rgb2gray(img)
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0
    if img.shape != (subset_size, subset_size):
        raise ValueError(f"{path} size {img.shape} ≠ subset_size={subset_size}")

    H, W = img.shape
    # referencia con ruido (y atenuación si procede)
    ref = img.copy()
    if attenuation_amp > 0:
        mref = attenuation_mask(H, W, attenuation_amp, rng, sigma=32.0)
        ref = ref * mref
    ns_ref = float(noise_scale * rng.uniform(0.9, 1.1))
    ref_noisy = add_poisson_noise(ref, scale=ns_ref, rng=rng)
    ref_noisy = add_read_noise(ref_noisy, gaussian_read_std, rng=rng)
    ref_pad = np.pad(ref_noisy, 2, mode='reflect')
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    # Normaliza probabilidades de modos
    probs = np.array([p_global, p_local, p_mix], dtype=np.float64)
    if probs.sum() <= 0:
        probs[:] = [0.3, 0.7, 0.0]
    probs = probs / probs.sum()

    # Archivos de anotación locales (por worker)
    ann_train_path = Path(ann_tmp_dir) / f"train_{idx:06d}.txt"
    ann_test_path  = Path(ann_tmp_dir) / f"test_{idx:06d}.txt"
    lines_train = []
    lines_test  = []

    # === TRAIN: mezcla global/local/mix
    for l in range(1, n_train + 1):
        mode = rng.choice(["global", "local", "mix"], p=probs)

        if mode == "global":
            dx, dy = generate_global_displacement(
                subset_size, global_sigma, global_amp, max_disp_global, rng=rng,
                affine_prob=0.5, affine_mag=0.003
            )
        elif mode == "local":
            dx, dy = generate_single_patch_displacement(
                patch_sizes, subset_size, max_disp=max_disp_local, rng=rng, blend_sigma=mode_blend_sigma
            )
        else:  # mix
            dx_g, dy_g = generate_global_displacement(
                subset_size, global_sigma, global_amp, max_disp_global, rng=rng,
                affine_prob=0.5, affine_mag=0.003
            )
            dx_l, dy_l = generate_multi_patch_displacement(
                patch_sizes, subset_size, k_min=mix_kmin, k_max=mix_kmax,
                max_disp=max_disp_local, rng=rng, blend_sigma=1.0
            )
            dx = dx_g + dx_l
            dy = dy_g + dy_l
            # clip final por seguridad
            lim = max(max_disp_local, max_disp_global)
            dx = np.clip(dx, -lim, lim).astype(np.float32)
            dy = np.clip(dy, -lim, lim).astype(np.float32)

        # deformar y añadir ruido/atenuación para la Def
        bd = apply_deformation(ref_pad, X, Y, dx, dy)
        if attenuation_amp > 0:
            mdef = attenuation_mask(H, W, attenuation_amp, rng, sigma=32.0)
            bd = bd * mdef
        ns = float(noise_scale * rng.uniform(0.9, 1.1))
        bd = add_poisson_noise(bd, scale=ns, rng=rng)
        bd = add_read_noise(bd, gaussian_read_std, rng=rng)

        files = save_data(Path(train_dir), idx, l, ref_noisy, bd, dx, dy,
                          fmt=save_fmt, compress=compress)
        lines_train.append(",".join(files))

    # === TEST: parches locales (1 parche) con misma escala de ruido nominal ===
    for l in range(1, n_test + 1):
        dx, dy = generate_single_patch_displacement(
            patch_sizes, subset_size, max_disp=max_disp_local, rng=rng, blend_sigma=mode_blend_sigma
        )
        bd = apply_deformation(ref_pad, X, Y, dx, dy)
        if attenuation_amp > 0:
            mdef = attenuation_mask(H, W, attenuation_amp, rng, sigma=32.0)
            bd = bd * mdef
        ns = float(noise_scale * rng.uniform(0.9, 1.1))
        bd = add_poisson_noise(bd, scale=ns, rng=rng)
        bd = add_read_noise(bd, gaussian_read_std, rng=rng)

        files = save_data(Path(test_dir), idx, l, ref_noisy, bd, dx, dy,
                          fmt=save_fmt, compress=compress)
        lines_test.append(",".join(files))

    # Escribir anotaciones locales (un archivo por referencia)
    ann_train_path.parent.mkdir(parents=True, exist_ok=True)
    if lines_train:
        ann_train_path.write_text("\n".join(sorted(lines_train)) + "\n")
    if lines_test:
        ann_test_path.write_text("\n".join(sorted(lines_test)) + "\n")

    # logs discretos
    if (k_ref + 1) % max(1, n_refs_total // 15) == 0:
        print(f"  · {k_ref + 1}/{n_refs_total} refs procesadas...")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Data augmentation speckle (v3: mix, afin, HPC-friendly)")
    p.add_argument('--refs_dir',    required=True)
    p.add_argument('--out_dir',     required=True)
    p.add_argument('--max_train',   type=int,   default=80000,   help="total de ejemplos train deseados")
    p.add_argument('--n_test',      type=int,   default=8)
    p.add_argument('--subset_size', type=int,   default=256)
    p.add_argument('--patch_sizes', nargs='+', type=int, default=[6, 8, 12, 18])

    # Probabilidades de modos
    p.add_argument('--p_global',     type=float, default=0.30)
    p.add_argument('--p_local',      type=float, default=0.70)
    p.add_argument('--p_mix',        type=float, default=0.00, help="global + varios parches locales")

    # Global
    p.add_argument('--global_sigma', type=float, default=7.5, help="σ (px) del blur para global")
    p.add_argument('--global_amp',   type=float, default=0.4, help="amplitud nominal (px) global")
    p.add_argument('--max_disp_global', type=float, default=0.6)

    # Local
    p.add_argument('--max_disp_local',  type=float, default=0.6)
    p.add_argument('--local_blend_sigma', type=float, default=0.0, help="suavizado global leve del parche (px)")
    p.add_argument('--local_patches_min', type=int, default=2, help="solo usado en modo mix")
    p.add_argument('--local_patches_max', type=int, default=4)

    # Ruido / atenuación
    p.add_argument('--noise_scale',  type=float, default=1.0)
    p.add_argument('--gaussian_read_std', type=float, default=0.0, help="ruido de lectura (0..255)")
    p.add_argument('--attenuation_amp',   type=float, default=0.0, help="amplitud de máscara multiplicativa suave")

    # HPC / I/O
    p.add_argument('--workers', type=int, default=0, help="nº procesos (0 = cpu_count())")
    p.add_argument('--save-format', choices=['csv','npy'], default='csv')
    p.add_argument('--compress', action='store_true', help="CSV .gz (reduce disco; más CPU)")

    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()

    # prepara dataset
    dataset = Path(args.out_dir)
    if dataset.exists():
        for child in dataset.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    dataset.mkdir(parents=True, exist_ok=True)

    refs = sorted(Path(args.refs_dir).glob("*.tif"))
    total_refs = len(refs)
    if total_refs == 0:
        print("❌ No se hallaron referencias .tif en", args.refs_dir)
        return

    per_ref = max(1, math.ceil(args.max_train / total_refs))
    train_dir = dataset / "Train_Data"
    test_dir  = dataset / "Test_Data"
    ann_tmp_dir = dataset / "_ann_tmp"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    ann_tmp_dir.mkdir(exist_ok=True)

    print(f"🧪 Generando: {per_ref} train / {args.n_test} test por ref × {total_refs} refs "
          f"(~{per_ref*total_refs} train totales)…")

    # Empaqueta args por worker (seed_base via args.seed)
    args_list = []
    for k, pth in enumerate(refs):
        args_list.append((
            str(pth), str(train_dir), str(test_dir),
            per_ref, args.n_test,
            int(args.subset_size), list(map(int, args.patch_sizes)),
            str(ann_tmp_dir),
            float(args.p_global), float(args.p_local), float(args.p_mix),
            float(args.global_sigma), float(args.global_amp), float(args.noise_scale),
            float(args.max_disp_local), float(args.max_disp_global),
            int(args.seed), k, total_refs,
            float(args.local_blend_sigma),
            int(args.local_patches_min), int(args.local_patches_max),
            float(args.gaussian_read_std), float(args.attenuation_amp),
            args.save_format, bool(args.compress)
        ))

    nproc = (cpu_count() if args.workers == 0 else max(1, int(args.workers)))
    with Pool(nproc) as pool:
        for _ in pool.imap_unordered(process_image, args_list):
            pass

    # Fusionar anotaciones sin Manager
    train_lines, test_lines = [], []
    for f in sorted(ann_tmp_dir.glob("train_*.txt")):
        train_lines.extend(f.read_text().strip().splitlines())
    for f in sorted(ann_tmp_dir.glob("test_*.txt")):
        test_lines.extend(f.read_text().strip().splitlines())

    (dataset / "Train_annotations.csv").write_text("\n".join(sorted(train_lines)) + ("\n" if train_lines else ""))
    (dataset / "Test_annotations.csv").write_text("\n".join(sorted(test_lines)) + ("\n" if test_lines else ""))

    # Limpieza temporal
    shutil.rmtree(ann_tmp_dir, ignore_errors=True)

    print(f"✅ Generadas {len(train_lines)} TRAIN y {len(test_lines)} TEST")
    print(f"   -> {train_dir}\n   -> {test_dir}")


if __name__ == "__main__":
    main()
