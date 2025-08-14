#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# run_inference.sh — Wrapper para Inference_Pipeline.py
# -----------------------------------------------------
# USO:
#   ./run_inference.sh infer   --data_dir DATA --save_dir OUT --ckpt model.pth [--cal-mode ...]
#   ./run_inference.sh metrics --pred_dir OUT  --gt_dir GT   --out_csv res.csv
#   ./run_inference.sh both    --data_dir DATA --save_dir OUT --ckpt model.pth --gt_dir GT --out_csv res.csv [--cal-mode ...]
#
set -euo pipefail

###############################################################################
# 0) Config por env (opcional)
###############################################################################
RECREATE_ENV=${RECREATE_ENV:-0}   # 1 = recrear venv cada ejecución; 0 = reutilizar
PYTHON_BIN="${PYTHON_BIN:-}"      # e.g., /apps/python3/3.9.2/bin/python3

###############################################################################
# 1) Módulos del sistema
###############################################################################
echo "🔧 Cargando módulos python3/3.9.2 y pytorch/1.10.0"
module load python3/3.9.2 pytorch/1.10.0

###############################################################################
# 2) Entorno virtual (usa site-packages del módulo → ve PyTorch)
###############################################################################
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VENV_DIR="$PROJECT_DIR/infenv"

if [[ "$RECREATE_ENV" == "1" && -d "$VENV_DIR" ]]; then
  echo "🔄 Eliminando entorno previo: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi
if [[ ! -d "$VENV_DIR" ]]; then
  echo "🔧 Creando virtualenv en $VENV_DIR (con --system-site-packages)"
  python3 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

###############################################################################
# 3) Dependencias mínimas (PyTorch viene del módulo del clúster)
###############################################################################
echo "📦 Instalando dependencias (compatibles con PyTorch 1.10)"
pip install --quiet --upgrade pip
pip install --quiet "numpy==1.23.5" tqdm

# Evitar sobre-suscripción de hilos BLAS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Elegir Python
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi
echo "🐍 Usando intérprete: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import torch
print("✅ torch versión:", torch.__version__)
PY

###############################################################################
# 4) Script de inferencia
###############################################################################
SCRIPT_PY="$PROJECT_DIR/Inference_Pipeline.py"
if [[ ! -f "$SCRIPT_PY" ]]; then
  echo "❌ No encuentro Inference_Pipeline.py en $PROJECT_DIR"
  deactivate
  exit 3
fi
echo "📄 Script: $SCRIPT_PY"

###############################################################################
# 5) Ayuda / Uso
###############################################################################
if [[ $# -lt 1 ]]; then
  cat >&2 <<'USAGE'
Uso:
  run_inference.sh {infer|metrics|both} [flags]

Flags comunes soportados por Inference_Pipeline.py:
  --cal-mode {none,A,B,B0}
  --gain GAIN            (A)
  --ax AX --ay AY [--bx BX --by BY]  (B/B0)
  --save-conf
  --no_phase_dark
  --neutralize-calib
  --min_mag VAL
  --phase_scale_x Sx --phase_scale_y Sy
  --wavelength LAMBDA
  --dark_ksize K
  --save_stats

Ejemplos:
  # Inferencia + métricas, sin fase, con máscara y neutralizando calibración
  ./run_inference.sh both \
    --data_dir /g/data/.../Test_Set_Output \
    --save_dir /g/data/.../Pred \
    --ckpt     /g/data/.../model_best.pth.tar \
    --gt_dir   /g/data/.../Test_Set_Output \
    --cal-mode none --no_phase_dark --neutralize-calib \
    --min_mag 0.05 --save-conf \
    --out_csv  /g/data/.../Pred/metrics_masked.csv
USAGE
  deactivate
  exit 1
fi

###############################################################################
# 6) Ejecución
###############################################################################
MODE="$1"; shift
case "$MODE" in
  infer|metrics|both) ;;
  *) echo "❌ Modo inválido: $MODE (usa infer|metrics|both)"; deactivate; exit 2 ;;
esac

echo "🚀 Ejecutando: $MODE $*"
"$PYTHON_BIN" "$SCRIPT_PY" "$MODE" "$@"

deactivate
echo "✅ Listo"
