#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# run_generate_test_set.sh
# ------------------------
# Crea/usa un venv, instala deps y lanza generate_test_set.py.
#
# Uso:
#   ./run_generate_test_set.sh [--n_aug N] [--subset_size S] [...]
#   (cualquier flag adicional se pasa al script Python)

set -Eeuo pipefail

# ===== 0) Opciones por entorno =====
RECREATE_ENV=${RECREATE_ENV:-0}     # 1=recrear venv cada vez, 0=reutilizar
PYTHON_BIN="${PYTHON_BIN:-}"        # e.g. /apps/python3/3.9.2/bin/python3

# ===== 1) Rutas (ajusta si hace falta) =====
REFS_DIR="/g/data/w09/cg2265/LorcasNet/Data_augmentation/Train_References"
OUT_DIR="/g/data/w09/cg2265/LorcasNet/Test_Set_Output"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PYTHON_SCRIPT="/g/data/w09/cg2265/LorcasNet/Data_augmentation/ generate_test_set.py"  # <- sin espacio :)

VENV_DIR="${SCRIPT_DIR}/venv_testgen"

# ===== 2) Cargar módulo (opcional en HPC) =====
if command -v module &>/dev/null; then
  module load python3/3.9.2 || true
fi

# ===== 3) Helpers =====
die() { echo "Error: $*" >&2; exit 1; }
on_exit() { set +u; [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate || true; }
trap on_exit EXIT

# ===== 4) Comprobaciones =====
[[ -f "$PYTHON_SCRIPT" ]] || die "No encuentro el script: $PYTHON_SCRIPT"
[[ -d "$REFS_DIR"    ]]   || die "No encuentro la carpeta de referencias: $REFS_DIR"

# ===== 5) Virtualenv =====
if [[ "$RECREATE_ENV" == "1" && -d "$VENV_DIR" ]]; then
  echo "🔄 Eliminando venv previo: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi
if [[ ! -d "$VENV_DIR" ]]; then
  echo "🔧 Creando entorno virtual en: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Elegir Python dentro del venv
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi
echo "🐍 Usando intérprete: $PYTHON_BIN"

# ===== 6) Dependencias =====
echo "📦 Instalando dependencias (compatibles con entornos HPC)…"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install "numpy>=1.21,<2" "scipy>=1.9,<1.12" "scikit-image>=0.20" tqdm

# Evitar sobre-suscripción de hilos BLAS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ===== 7) Limpiar salida y ejecutar =====
echo "🧹 Limpiando directorio de salida: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "🚀 Ejecutando generador:"
set -x
"$PYTHON_BIN" "$PYTHON_SCRIPT" \
  --refs_dir "$REFS_DIR" \
  --out_dir  "$OUT_DIR" \
  "$@"
set +x

echo "✅ Conjunto de prueba generado: $OUT_DIR"
