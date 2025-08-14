#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage import io
from skimage.color import rgb2gray


def generate_smooth_displacement_field(subset_size: int, sigma: float, amplitude: float,
                                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera un campo de desplazamiento continuo de tamaño subset_size:
    - ruido gaussiano N(0, amplitude) -> suavizado gaussiano con sigma (px)
    """
    fx = rng.normal(0.0, amplitude, (subset_size, subset_size))
    fy = rng.normal(0.0, amplitude, (subset_size, subset_size))
    dx = gaussian_filter(fx, sigma=sigma, mode='reflect')
    dy = gaussian_filter(fy, sigma=sigma, mode='reflect')
    return dx, dy


def rescale_max_abs(dx: np.ndarray, dy: np.ndarray, max_disp: float) -> tuple[np.ndarray, np.ndarray]:
    """Reescala (dx,dy) para que su máximo absoluto sea <= max_disp (si max_disp > 0)."""
    if max_disp is None or max_disp <= 0:
        return dx, dy
    m = max(float(np.max(np.abs(dx))), float(np.max(np.abs(dy))), 1e-12)
    s = max_disp / m
    return dx * s, dy * s


def apply_poisson_noise(img: np.ndarray, scale: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Aplica ruido Poisson preservando rango [0,255].
    Si scale <= 0, devuelve la imagen tal cual (clipeada al rango).
    """
    if scale <= 0:
        return np.clip(img, 0, 255)
    if rng is None:
        rng = np.random.default_rng()
    s = img * scale
    out = img.copy()
    mask = (s >= 0) & np.isfinite(s)
    # Sólo donde es válido
    out[mask] = rng.poisson(s[mask]).astype(np.float64) / scale
    return np.clip(out, 0, 255)


def warp_dense(ref_noisy: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """
    Warpeo denso de imagen usando map_coordinates, con padding reflect y margen de 2 px.
    """
    H, W = ref_noisy.shape
    ref_pad = np.pad(ref_noisy, 2, mode="reflect")
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # Coordenadas (y, x) con desplazamiento y offset del padding
    yn = Y + dy + 2.0
    xn = X + dx + 2.0
    warped = map_coordinates(ref_pad, [yn, xn], order=3, mode='reflect')
    return warped


def save_outputs(out_dir: Path, idx: int, i: int, ref_noisy, bd, dx, dy) -> None:
    name_ref = out_dir / f"Ref{idx:03d}_{i:03d}.csv"
    name_def = out_dir / f"Def{idx:03d}_{i:03d}.csv"
    name_dx  = out_dir / f"Dispx{idx:03d}_{i:03d}.csv"
    name_dy  = out_dir / f"Dispy{idx:03d}_{i:03d}.csv"
    np.savetxt(name_ref, ref_noisy.astype(np.float32), delimiter=",", fmt="%.6f")
    np.savetxt(name_def, bd.astype(np.float32),       delimiter=",", fmt="%.6f")
    np.savetxt(name_dx,  dx.astype(np.float32),       delimiter=",", fmt="%.6f")
    np.savetxt(name_dy,  dy.astype(np.float32),       delimiter=",", fmt="%.6f")


def _extract_idx_from_name(stem: str, fallback: int) -> int:
    """Extrae el primer bloque de dígitos del nombre; si falla, usa fallback."""
    m = re.search(r'(\d+)', stem)
    return int(m.group(1)) if m else int(fallback)


def process_ref(args):
    path_str, out_dir, cfg, k_worker = args  # k_worker: índice estable por worker
    path = Path(path_str)
    img = io.imread(path_str)
    if img.ndim == 3:
        img = (rgb2gray(img) * 255.0)
    ref = img.astype(np.float64)

    # Chequeo de tamaño
    H, W = ref.shape
    if (H, W) != (cfg.subset_size, cfg.subset_size):
        raise ValueError(f"{path.name}: tamaño {ref.shape} ≠ subset_size={cfg.subset_size}")

    # Semilla reproducible por worker
    rng = np.random.default_rng(int(cfg.seed) + int(k_worker))

    # Identificador robusto
    idx = _extract_idx_from_name(path.stem, fallback=k_worker + 1)

    # Ruido sobre referencia
    ref_noisy = apply_poisson_noise(ref, cfg.noise_scale, rng=rng)

    for i in range(1, cfg.n_aug + 1):
        dx, dy = generate_smooth_displacement_field(cfg.subset_size, cfg.sigma, cfg.amplitude, rng)
        dx, dy = rescale_max_abs(dx, dy, getattr(cfg, 'max_disp', None))
        bd = warp_dense(ref_noisy, dx, dy)
        bd = apply_poisson_noise(bd, cfg.noise_scale, rng=rng)
        save_outputs(out_dir, idx, i, ref_noisy, bd, dx, dy)


def main():
    p = argparse.ArgumentParser(description="Augmentación: deformación suave densa para pruebas de métricas")
    p.add_argument('--refs_dir',    type=Path, required=True, help='Carpeta con Ref*.tif')
    p.add_argument('--out_dir',     type=Path, required=True, help='Directorio de salida (CSV)')
    p.add_argument('--n_aug',       type=int,   default=15,   help='Variaciones por referencia')
    p.add_argument('--subset_size', type=int,   default=256,  help='Tamaño H=W esperado')
    p.add_argument('--sigma',       type=float, default=4.0,  help='Suavizado gaussiano (px)')
    p.add_argument('--amplitude',   type=float, default=1.6,  help='STD inicial del ruido')
    p.add_argument('--max_disp',    type=float, default=0.0,  help='Máximo |disp| tras suavizar (0=sin límite)')
    p.add_argument('--noise_scale', type=float, default=1.0,  help='Escala para Poisson noise')
    p.add_argument('--seed',        type=int,   default=1234, help='Semilla base')
    p.add_argument('--workers',     type=int,   default=cpu_count(), help='Procesos paralelos')
    cfg = p.parse_args()

    # Limpiar directorio de salida
    if cfg.out_dir.exists():
        shutil.rmtree(cfg.out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    refs = sorted(cfg.refs_dir.glob('Ref*.tif'))
    if not refs:
        print(f"⚠️  No encontré referencias en {cfg.refs_dir}")
        return

    tasks = [(str(r), cfg.out_dir, cfg, i) for i, r in enumerate(refs)]
    with Pool(cfg.workers) as pool:
        pool.map(process_ref, tasks)

    print(f"✅ Generadas {len(refs)*cfg.n_aug} deformaciones y mapas de desplazamiento.")

    # Generar Metric_annotations.csv
    annos = []
    for f in sorted(os.listdir(cfg.out_dir)):
        if f.startswith('Ref') and f.endswith('.csv'):
            def_name = f.replace('Ref', 'Def')
            if (cfg.out_dir / def_name).exists():
                annos.append((f, def_name))
    with open(cfg.out_dir / 'Metric_annotations.csv', 'w') as fh:
        for ref_name, defn in annos:
            fh.write(f"{ref_name},{defn}\n")

    print("✅ Anotaciones guardadas en Metric_annotations.csv")


if __name__ == '__main__':
    main()
