#!/usr/bin/env python3
import os
import argparse
import shutil
import numpy as np
from pathlib import Path
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage import io
from multiprocessing import Pool, cpu_count, Manager
import random
from skimage.color import rgb2gray

def generate_single_patch_displacement(patch_sizes, subset_size, sigma=0.8):
    """Desplazamiento puntual: parche aleatorio con ruido gaussiano."""
    patch_size = random.choice(patch_sizes)
    x0 = random.randint(0, subset_size - patch_size)
    y0 = random.randint(0, subset_size - patch_size)
    disp_x = np.zeros((subset_size, subset_size), dtype=np.float64)
    disp_y = np.zeros((subset_size, subset_size), dtype=np.float64)
    fx = np.clip(np.random.normal(0, sigma, (patch_size, patch_size)), -3, 3)
    fy = np.clip(np.random.normal(0, sigma, (patch_size, patch_size)), -3, 3)
    disp_x[y0:y0+patch_size, x0:x0+patch_size] = fx
    disp_y[y0:y0+patch_size, x0:x0+patch_size] = fy
    return disp_x, disp_y

def generate_global_displacement(subset_size, sigma, amplitude):
    """Campo suave global: ruido gaussiano suave en toda la imagen."""
    dx = np.random.normal(0, sigma, (subset_size, subset_size))
    dy = np.random.normal(0, sigma, (subset_size, subset_size))
    dx = gaussian_filter(dx, sigma) * amplitude
    dy = gaussian_filter(dy, sigma) * amplitude
    return dx, dy

def apply_deformation(ref_noisy, ref_pad, X, Y, dx, dy):
    bd = ref_noisy.copy()
    mask = (dx != 0) | (dy != 0)
    yi, xi = np.where(mask)
    xi_new = X[yi, xi] + dx[yi, xi] + 2
    yi_new = Y[yi, xi] + dy[yi, xi] + 2
    vals = map_coordinates(ref_pad, [yi_new, xi_new], order=3, mode='reflect')
    bd[yi, xi] = vals
    return bd

def add_poisson_noise(img, scale=1.0):
    img_s = img * scale
    m = (img_s >= 0) & (~np.isnan(img_s))
    noisy = img.copy()
    noisy[m] = np.random.poisson(img_s[m]).astype(np.float64) / scale
    return np.clip(noisy, 0, 255)

def apply_attenuation(img, mask):
    att = 0.5 + 0.5 * random.random()
    img[mask] *= att
    return np.clip(img, 0, 255)

def save_data(out_dir, idx, l, ref_noisy, bd_noisy, dx, dy):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [
        f"Ref{idx:03d}_{l:03d}.csv",
        f"Def{idx:03d}_{l:03d}.csv",
        f"Dispx{idx:03d}_{l:03d}.csv",
        f"Dispy{idx:03d}_{l:03d}.csv"
    ]
    for name, arr in zip(names, (ref_noisy, bd_noisy, dx, dy)):
        np.savetxt(str(out_dir / name), arr, delimiter=",", fmt="%.6f")
    return names

def process_image(args):
    (path, train_dir, test_dir, n_train, n_test,
     subset_size, patch_sizes, train_ann, test_ann,
     p_global, p_local, global_sigma, global_amp, noise_scale) = args

    idx = int(Path(path).stem.split('_')[-1])
    img = io.imread(path)
    if img.ndim == 3:
        img = rgb2gray(img)
    img = img.astype(np.float64)

    # Referencia con ruido inicial
    ref_noisy = add_poisson_noise(img, scale=noise_scale)
    ref_pad = np.pad(ref_noisy, 2, mode='reflect')
    X, Y = np.meshgrid(np.arange(subset_size), np.arange(subset_size))

    # === Generación TRAIN con mezcla de deformaciones ===
    for l in range(1, n_train + 1):
        r = random.random()
        if r < p_global:
            dx, dy = generate_global_displacement(subset_size, global_sigma, global_amp)
        elif r < p_global + p_local:
            dx, dy = generate_single_patch_displacement(patch_sizes, subset_size)
        else:
            # parche local con más ruido
            dx, dy = generate_single_patch_displacement(patch_sizes, subset_size, sigma=2.0)
        bd = apply_deformation(ref_noisy, ref_pad, X, Y, dx, dy)
        bd = add_poisson_noise(bd, scale=noise_scale)
        bd = apply_attenuation(bd, (dx != 0) | (dy != 0))
        files = save_data(train_dir, idx, l, ref_noisy, bd, dx, dy)
        train_ann.append(files)

    # === Generación TEST (solo parches locales, igual escala de ruido) ===
    for l in range(1, n_test + 1):
        dx, dy = generate_single_patch_displacement(patch_sizes, subset_size)
        bd = apply_deformation(ref_noisy, ref_pad, X, Y, dx, dy)
        bd = add_poisson_noise(bd, scale=noise_scale)
        bd = apply_attenuation(bd, (dx != 0) | (dy != 0))
        files = save_data(test_dir, idx, l, ref_noisy, bd, dx, dy)
        test_ann.append(files)

def main():
    parser = argparse.ArgumentParser(description="Data augmentation rápido (v2)")
    parser.add_argument('--refs_dir',    required=True, help="Carpeta con I_ref_step_*.tif")
    parser.add_argument('--out_dir',     required=True, help="Root donde se creará Dataset")
    parser.add_argument('--max_train',   type=int,   default=5000, help="Máximo total de muestras train")
    parser.add_argument('--n_test',      type=int,   default=2,    help="Muestras test por imagen")
    parser.add_argument('--subset_size', type=int,   default=256)
    parser.add_argument('--patch_sizes', nargs='+', type=int, default=[4,8,16,32,64,128])
    parser.add_argument('--global_sigma', type=float, default=4.0, help="σ para campo global")
    parser.add_argument('--global_amp',   type=float, default=0.8, help="amplitud campo global")
    parser.add_argument('--p_global',     type=float, default=0.5, help="probabilidad deformación global")
    parser.add_argument('--p_local',      type=float, default=0.4, help="probabilidad parche local")
    parser.add_argument('--noise_scale',  type=float, default=1.0, help="escala Poisson noise")
    args = parser.parse_args()

    dataset = Path(args.out_dir)
    if dataset.exists():
        for child in dataset.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    dataset.mkdir(parents=True, exist_ok=True)

    refs = sorted(Path(args.refs_dir).glob("I_ref_step_*.tif"))
    total_refs = len(refs)
    if total_refs == 0:
        print("❌ No se hallaron referencias .tif en", args.refs_dir)
        return

    per_ref = max(1, args.max_train // total_refs)
    train_dir = dataset / "Train_Data"
    test_dir  = dataset / "Test_Data"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    mgr = Manager()
    train_ann = mgr.list()
    test_ann  = mgr.list()

    args_list = []
    for p in refs:
        args_list.append((
            str(p), train_dir, test_dir,
            per_ref, args.n_test,
            args.subset_size, args.patch_sizes,
            train_ann, test_ann,
            args.p_global, args.p_local,
            args.global_sigma, args.global_amp,
            args.noise_scale
        ))

    with Pool(cpu_count()) as pool:
        pool.map(process_image, args_list)

    # Guardar CSVs de anotaciones
    (dataset / "Train_annotations.csv").write_text(
        "\n".join(",".join(x) for x in sorted(train_ann))
    )
    (dataset / "Test_annotations.csv").write_text(
        "\n".join(",".join(x) for x in sorted(test_ann))
    )

    print(f"✅ Generadas {len(train_ann)} TRAIN y {len(test_ann)} TEST")
    print(f"   -> {train_dir}")
    print(f"   -> {test_dir}")

if __name__ == "__main__":
    main()
