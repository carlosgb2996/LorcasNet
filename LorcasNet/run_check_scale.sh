#!/bin/bash
# run_check_scale.sh
set -euo pipefail

###############################################################################
# 0) Ayuda rápida
###############################################################################
usage() {
  cat <<'EOF'
Uso:
  ./run_check_scale.sh [MODE] [i0] [i1] [--fresh-venv] [args extras...]

MODE: A | B | both | none   (por defecto: both)
i0,i1: índices para el par ejemplo (por defecto: 001 001)

Ejemplos:
  ./run_check_scale.sh both 001 001 --cal-a-global --cal-b0
  ./run_check_scale.sh A 010 007 --pred-dir /path/Pred
  ./run_check_scale.sh none --global-sample 500
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage; exit 0
fi

###############################################################################
# 1) Cargar módulos del sistema
###############################################################################
echo "🔧 Cargando módulos python3/3.9.2 y pytorch/1.10.0"
module load python3/3.9.2 pytorch/1.10.0

###############################################################################
# 2) Parseo básico y entorno virtual
###############################################################################
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VENV_DIR="$PROJECT_DIR/testenv"

MODE="${1:-both}"
I0="${2:-001}"
I1="${3:-001}"

# shift posicionales consumidos si existen
shift $(( $# >= 3 ? 3 : $# )) || true

FRESH_VENV=0
# Detecta flag --fresh-venv en args extras y lo elimina de la lista
EXTRA_ARGS=()
for a in "$@"; do
  if [[ "$a" == "--fresh-venv" ]]; then
    FRESH_VENV=1
  else
    EXTRA_ARGS+=("$a")
  fi
done

if [[ $FRESH_VENV -eq 1 && -d "$VENV_DIR" ]]; then
  echo "🔄 Eliminando entorno previo (por --fresh-venv): $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "🔧 Creando virtualenv en $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# un poco de higiene
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

###############################################################################
# 3) Instalar dependencias necesarias
###############################################################################
echo "📦 Instalando dependencias: numpy, scipy, tqdm, tensorboardX, packaging"
python -m pip install --upgrade pip setuptools wheel >/dev/null
python -m pip install "numpy==1.23.5" "scipy>=1.8,<1.12" tqdm tensorboardX==2.6 packaging >/dev/null

###############################################################################
# 4) Config por defecto y sanity checks
###############################################################################
BASE_DIR="/g/data/w09/cg2265/LorcasNet"
DEFAULT_PRED_DIR="$BASE_DIR/Pred"
DEFAULT_GT_DIR="$BASE_DIR/Test_Set_Output"
DEFAULT_CHKPT="$BASE_DIR/LorcasNet/results/20250808_152318/LorcasNet_bn,adamw,30ep,b16,lr0.0003/model_best.pth.tar"
ARCH="LorcasNet_bn"

# Permitir override desde EXTRA_ARGS (argparse del py acepta --pred-dir/--gt-dir/--chkpt)
PRED_DIR="$DEFAULT_PRED_DIR"
GT_DIR="$DEFAULT_GT_DIR"
CHKPT="$DEFAULT_CHKPT"

# Extra: si usuario sobreescribe en EXTRA_ARGS, no tocaremos nada.

# Sanity: GT siempre requerido
if [[ ! -d "$GT_DIR" && " ${EXTRA_ARGS[*]-} " != *"--gt-dir"* ]]; then
  echo "❌ No existe GT_DIR por defecto: $GT_DIR (usa --gt-dir ...)"
  exit 1
fi

# Sanity: Pred necesario para modos A/B/both (para métricas por par)
if [[ "$MODE" != "none" && ! -d "$PRED_DIR" && " ${EXTRA_ARGS[*]-} " != *"--pred-dir"* ]]; then
  echo "⚠️  Pred_DIR por defecto no existe: $PRED_DIR (continuará, pero se omitirán métricas de ejemplo)"
fi

# Sanity: Checkpoint solo si hacemos forward (opcional; el script Python no lo exige)
if [[ "$MODE" != "none" && ! -f "$CHKPT" && " ${EXTRA_ARGS[*]-} " != *"--chkpt"* ]]; then
  echo "ℹ️  Checkpoint por defecto no encontrado: $CHKPT (se omitirá forward/identidad)"
  CHKPT=""
fi

###############################################################################
# 5) Construcción del comando
###############################################################################
CMD=( python3 "$PROJECT_DIR/check_scale_and_slopes.py"
      --mode "$MODE"
      --gt-dir "$GT_DIR"
      --i0 "$I0" --i1 "$I1"
    )

# Añade pred-dir si existe (o el usuario pasó uno en EXTRA_ARGS)
if [[ -d "$PRED_DIR" ]]; then
  CMD+=( --pred-dir "$PRED_DIR" )
fi

# Añade chkpt/arch si el archivo existe (o lo pasarán en EXTRA_ARGS)
if [[ -n "$CHKPT" && -f "$CHKPT" ]]; then
  CMD+=( --chkpt "$CHKPT" --arch "$ARCH" )
fi

# Flags por defecto útiles (puedes quitarlos si prefieres setearlos a mano)
# - Ganancia A desde muestreo global
# - B0 = sin intercepto (más estable si hay sesgo pequeño)
if [[ "$MODE" == "A" || "$MODE" == "both" ]]; then
  CMD+=( --cal-a-global )
fi
if [[ "$MODE" == "B" || "$MODE" == "both" ]]; then
  CMD+=( --cal-b0 )
fi

# Pasa el resto de flags tal cual (e.g., --global-sample 500, etc.)
CMD+=( "${EXTRA_ARGS[@]}" )

echo "🚀 Ejecutando: ${CMD[*]}"
"${CMD[@]}"
