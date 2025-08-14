#!/usr/bin/env bash
# run_augment_data_v3.sh — Genera dataset sintético speckle (global/local/mix), HPC-friendly
set -euo pipefail

###############################################################################
# 0) (Opcional) Cargar módulo de Python en HPC
###############################################################################
if command -v module &>/dev/null; then
  echo "🔧 Cargando módulo python3/3.9.2 (si existe)…"
  module load python3/3.9.2 || true
fi

# Evitar sobre-subscription cuando usamos multiprocessing + BLAS
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

###############################################################################
# 1) Rutas y parámetros (override con env vars si quieres)
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$(command -v python3 || command -v python)"
AUG_SCRIPT="${AUG_SCRIPT:-$SCRIPT_DIR/augment_data_v3.py}"

# Rutas
REFS_DIR="${REFS_DIR:-/g/data/w09/cg2265/LorcasNet/Data_augmentation/Train_References_1}"
OUT_DIR="${OUT_DIR:-/g/data/w09/cg2265/LorcasNet/Dataset}"

# Tamaños / dataset
MAX_TRAIN="${MAX_TRAIN:-80000}"     # total de train aprox
N_TEST="${N_TEST:-8}"               # por referencia
SUBSET="${SUBSET:-256}"
PATCH_SIZES="${PATCH_SIZES:-6 8 12 18}"

# Probabilidades de modos
P_GLOBAL="${P_GLOBAL:-0.25}"
P_LOCAL="${P_LOCAL:-0.55}"
P_MIX="${P_MIX:-0.20}"              # nuevo: global + multi-parche

# Global
GLOBAL_SIGMA="${GLOBAL_SIGMA:-7.5}"
GLOBAL_AMP="${GLOBAL_AMP:-0.4}"
MAX_DISP_GLOBAL="${MAX_DISP_GLOBAL:-0.6}"

# Local
MAX_DISP_LOCAL="${MAX_DISP_LOCAL:-0.6}"
LOCAL_BLEND_SIGMA="${LOCAL_BLEND_SIGMA:-0.8}"   # suaviza borde del parche
LOCAL_PATCHES_MIN="${LOCAL_PATCHES_MIN:-2}"     # usado en mix
LOCAL_PATCHES_MAX="${LOCAL_PATCHES_MAX:-4}"

# Ruido / atenuación
NOISE_SCALE="${NOISE_SCALE:-1.0}"
GAUSSIAN_READ_STD="${GAUSSIAN_READ_STD:-0.5}"   # 0 = desactivar
ATTENUATION_AMP="${ATTENUATION_AMP:-0.05}"      # 0 = desactivar

# HPC / I/O
WORKERS="${WORKERS:-0}"             # 0 = auto (cpu_count)
SAVE_FORMAT="${SAVE_FORMAT:-csv}"   # csv | npy
COMPRESS="${COMPRESS:-false}"       # true -> CSV.gz

SEED="${SEED:-1234}"

###############################################################################
# 2) Preparar entorno mínimo de Python
###############################################################################
echo "🐍 Usando intérprete: $PYTHON"
"$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$PYTHON" -m pip install --upgrade --user pip setuptools wheel
# Dependencias: numpy/scipy (filtros/BLAS), scikit-image/tifffile (TIFFs)
"$PYTHON" -m pip install --user "numpy>=1.21" "scipy>=1.9,<1.12" "scikit-image>=0.20" tifffile

###############################################################################
# 3) Comprobaciones y salida
###############################################################################
if [[ ! -f "$AUG_SCRIPT" ]]; then
  echo "❌ No encuentro el generador: $AUG_SCRIPT"
  echo "   Asegúrate de haber guardado el archivo como 'augment_data_v3.py' en $SCRIPT_DIR"
  exit 1
fi

echo "🧹 Limpiando dataset: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

###############################################################################
# 4) Ejecutar data augmentation
###############################################################################
cd "$SCRIPT_DIR"

echo "⚙️  Ejecutando augmentation con modos global/local/mix…"
echo "   Refs: $REFS_DIR"
echo "   Out : $OUT_DIR"
echo "   Formato: $SAVE_FORMAT  |  Compresión: $COMPRESS  |  Workers: $WORKERS"

set -x
"$PYTHON" "$AUG_SCRIPT" \
  --refs_dir            "$REFS_DIR" \
  --out_dir             "$OUT_DIR" \
  --max_train           "$MAX_TRAIN" \
  --n_test              "$N_TEST" \
  --subset_size         "$SUBSET" \
  --patch_sizes         $PATCH_SIZES \
  --p_global            "$P_GLOBAL" \
  --p_local             "$P_LOCAL" \
  --p_mix               "$P_MIX" \
  --global_sigma        "$GLOBAL_SIGMA" \
  --global_amp          "$GLOBAL_AMP" \
  --max_disp_global     "$MAX_DISP_GLOBAL" \
  --max_disp_local      "$MAX_DISP_LOCAL" \
  --local_blend_sigma   "$LOCAL_BLEND_SIGMA" \
  --local_patches_min   "$LOCAL_PATCHES_MIN" \
  --local_patches_max   "$LOCAL_PATCHES_MAX" \
  --noise_scale         "$NOISE_SCALE" \
  --gaussian_read_std   "$GAUSSIAN_READ_STD" \
  --attenuation_amp     "$ATTENUATION_AMP" \
  --workers             "$WORKERS" \
  --save-format         "$SAVE_FORMAT" \
  $( [[ "$COMPRESS" == "true" ]] && echo "--compress" ) \
  --seed                "$SEED"
set +x

echo "✅ Augmentation terminado:"
echo "   • Train: $OUT_DIR/Train_Data/"
echo "   • Test : $OUT_DIR/Test_Data/"
echo "   • Índices: $OUT_DIR/Train_annotations.csv  |  $OUT_DIR/Test_annotations.csv"
