#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage import io
from skimage.color import rgb2gray


def generate_smooth_displacement_field(subset_size, sigma, amplitude):
    """
    Genera un campo de desplazamiento continuo de tamaño subset_size con suavizado gaussiano.
    """
    fx = np.random.normal(0, amplitude, (subset_size, subset_size))
    fy = np.random.normal(0, amplitude, (subset_size, subset_size))
    dx = gaussian_filter(fx, sigma=sigma, mode='reflect')
    dy = gaussian_filter(fy, sigma=sigma, mode='reflect')
    return dx, dy


def apply_poisson_noise(img, scale=1.0):
    """
    Aplica ruido Poisson a la imagen.
    Si scale <= 0, no aplica ruido y retorna la imagen original.
    """
    if scale <= 0:
        # No aplicar ruido cuando la escala es cero o negativa
        return np.clip(img, 0, 255)
    s = img * scale
    mask = (s >= 0) & (~np.isnan(s))
    out = img.copy()
    # Generar Poisson solo donde sea válido
    out[mask] = np.random.poisson(s[mask]).astype(np.float64) / scale
    return np.clip(out, 0, 255)(out, 0, 255)


def apply_deformation(ref_noisy, ref_pad, X, Y, dx, dy):
    """
    Aplica deformación en todo el fondo sin máscara.
    """
    mask_def = (dx != 0) | (dy != 0)
    idx = np.where(mask_def)
    xn = X[idx] + dx[idx] + 2
    yn = Y[idx] + dy[idx] + 2
    vals = map_coordinates(ref_pad, [yn, xn], order=3, mode='reflect')
    out = ref_noisy.copy()
    out[idx] = vals
    return out


def save_outputs(out_dir: Path, idx: int, i: int, ref_noisy, bd, dx, dy):
    name_ref = out_dir / f"Ref{idx:03d}_{i:03d}.csv"
    name_def = out_dir / f"Def{idx:03d}_{i:03d}.csv"
    name_dx  = out_dir / f"Dispx{idx:03d}_{i:03d}.csv"
    name_dy  = out_dir / f"Dispy{idx:03d}_{i:03d}.csv"
    np.savetxt(name_ref, ref_noisy, delimiter=",", fmt="%.6f")
    np.savetxt(name_def, bd, delimiter=",", fmt="%.6f")
    np.savetxt(name_dx,  dx, delimiter=",", fmt="%.6f")
    np.savetxt(name_dy,  dy, delimiter=",", fmt="%.6f")


def process_ref(args):
    path, out_dir, cfg = args
    img = io.imread(path)
    if img.ndim == 3:
        img = rgb2gray(img) * 255
    ref = img.astype(np.float64)
    idx = int(Path(path).stem.lstrip("Ref"))

    # Ruido y padding
    ref_noisy = apply_poisson_noise(ref, cfg.noise_scale)
    ref_pad = np.pad(ref_noisy, 2, mode="reflect")
    X, Y = np.meshgrid(np.arange(cfg.subset_size), np.arange(cfg.subset_size))

    for i in range(1, cfg.n_aug + 1):
        dx, dy = generate_smooth_displacement_field(
            subset_size=cfg.subset_size,
            sigma=cfg.sigma,
            amplitude=cfg.amplitude
        )
        bd = apply_deformation(ref_noisy, ref_pad, X, Y, dx, dy)
        bd = apply_poisson_noise(bd, cfg.noise_scale)
        save_outputs(out_dir, idx, i, ref_noisy, bd, dx, dy)


def main():
    p = argparse.ArgumentParser(description="Augmentación: deformación suave para todo el fondo")
    p.add_argument('--refs_dir',    type=Path, required=True, help='Carpeta con RefXXX.tif')
    p.add_argument('--out_dir',     type=Path, required=True, help='Directorio donde guardar los CSV')
    p.add_argument('--n_aug',       type=int,   default=15,   help='Variaciones por cada referencia')
    p.add_argument('--subset_size', type=int,   default=256,  help='Tamaño de la ventana')
    p.add_argument('--sigma',       type=float, default=4.0,  help='Suavizado gaussiano (pixeles)')
    p.add_argument('--amplitude',   type=float, default=1.6,  help='Desviación inicial del ruido')
    p.add_argument('--noise_scale', type=float, default=1.0,  help='Escala para Poisson noise')
    p.add_argument('--workers',     type=int,   default=cpu_count(), help='Procesos paralelos')
    cfg = p.parse_args()

    # Limpiar directorio de salida
    if cfg.out_dir.exists(): shutil.rmtree(cfg.out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    refs = sorted(cfg.refs_dir.glob('Ref*.tif'))
    if not refs:
        print(f"⚠️  No encontré referencias en {cfg.refs_dir}")
        return

    tasks = [(str(r), cfg.out_dir, cfg) for r in refs]
    with Pool(cfg.workers) as pool:
        pool.map(process_ref, tasks)

    print(f"✅ Generadas {len(refs)*cfg.n_aug} deformaciones y mapas de desplazamiento.")
    # Generar Metric_annotations.csv
    annos = []
    for f in sorted(os.listdir(cfg.out_dir)):
        if f.startswith('Ref') and f.endswith('.csv'):
            idx = f[len('Ref'):-len('.csv')]
            def_name = f.replace('Ref', 'Def')
            if os.path.exists(cfg.out_dir/def_name): annos.append((f, def_name))
    with open(cfg.out_dir/'Metric_annotations.csv','w') as f:
        for ref_name, defn in annos:
            f.write(f"{ref_name},{defn}\n")

    print("✅ Anotaciones guardadas en Metric_annotations.csv")

if __name__=='__main__':
    main()
