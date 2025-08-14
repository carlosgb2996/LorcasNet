#!/usr/bin/env bash
# run_train.sh — Entrena LorcasNet con PyTorch DDP + virtualenv
set -euo pipefail

###############################################################################
# 1) Parámetros del proyecto
###############################################################################
PROJECT_DIR="/g/data/w09/cg2265/LorcasNet/LorcasNet"
DATA_DIR="/g/data/w09/cg2265/LorcasNet"        # raíz que contiene Dataset/
NGPUS=${NGPUS:-4}                              # nº GPUs por nodo
NCPU=${NCPU:-8}                                # dataloader workers por GPU
ENV_DIR="$PROJECT_DIR/env"
LOG_DIR="$PROJECT_DIR/logs"

# Si NO reanudamos, usar carpeta nueva; si reanudamos, recalcularemos más abajo
OUT_DIR_DEFAULT="$PROJECT_DIR/results/$(date +%Y%m%d_%H%M%S)"

# Hiperparámetros clave
EPOCHS=${EPOCHS:-60}
BATCH=${BATCH:-16}
LR=${LR:-3e-4}
LAMBDA_CONF=${LAMBDA_CONF:-0.1}
P_IDENTITY=${P_IDENTITY:-0.10}
SEED=${SEED:-42}

# Reanudar (opcional)
RESUME_FROM="${RESUME_FROM:-}"

# Control del entorno (0 = reutiliza venv, 1 = lo recrea)
RECREATE_ENV=${RECREATE_ENV:-0}

###############################################################################
# 2) Entorno de ejecución
###############################################################################
echo "🔧 Cargando módulos python3/3.9.2 y pytorch/1.10.0"
module load python3/3.9.2 pytorch/1.10.0

# Virtualenv con acceso a site-packages del módulo (para ver torch del cluster)
if [[ "$RECREATE_ENV" == "1" && -d "$ENV_DIR" ]]; then
  echo "🔄 Eliminando venv existente: $ENV_DIR"
  rm -rf "$ENV_DIR"
fi
if [[ ! -d "$ENV_DIR" ]]; then
  python3 -m venv --system-site-packages "$ENV_DIR"
fi
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

echo "📦 Instalando dependencias en el venv"
pip install --upgrade pip
pip install --quiet --upgrade "numpy<2" pandas tensorboardX tqdm

echo "✅ numpy versión: $(python -c 'import numpy as np; print(np.__version__)')"
python - <<'PY'
import torch
print("✅ torch versión:", torch.__version__)
PY

###############################################################################
# 3) Directorios de salida
###############################################################################
mkdir -p "$LOG_DIR"

# Si reanudamos, usar el directorio del checkpoint; si no, crear uno nuevo
if [[ -n "$RESUME_FROM" ]]; then
  OUT_DIR="$(dirname "$RESUME_FROM")"
  echo "🔄 Reanudando entrenamiento — OUT_DIR: $OUT_DIR"
else
  OUT_DIR="$OUT_DIR_DEFAULT"
  mkdir -p "$OUT_DIR"
fi

###############################################################################
# 4) Variables NCCL / OpenMP para estabilidad
###############################################################################
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((12000 + RANDOM % 20000))
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1
export NCCL_BLOCKING_WAIT=1
# Si tu nodo da guerra con IB/SHM, puedes descomentar:
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_P2P_DISABLE=1

###############################################################################
# 5) Entrenamiento distribuido
###############################################################################
cd "$PROJECT_DIR"

if command -v torchrun >/dev/null 2>&1; then
  RUNNER=(torchrun --nnodes 1 --nproc_per_node "$NGPUS")
else
  RUNNER=(python -m torch.distributed.run --nnodes 1 --nproc_per_node "$NGPUS")
fi

echo "🚀 Iniciando entrenamiento distribuido en $NGPUS GPU(s) — logs en $LOG_DIR"
set -o pipefail
CMD=( "${RUNNER[@]}" Train.py
  --arch LorcasNet_bn
  --solver adamw
  --epochs "$EPOCHS"
  --batch-size "$BATCH"
  --lr "$LR"
  --weight-decay 1e-4
  --bias-decay 0
  --multiscale-weights 0.32 0.08 0.02 0.01 0.005
  --lambda-conf "$LAMBDA_CONF"
  --p-identity "$P_IDENTITY"
  --workers "$NCPU"
  --print-freq 50
  --save-path "$OUT_DIR"
  --data-dir  "$DATA_DIR"
  --seed "$SEED"
)

# Añadir --resume-from si está definido
if [[ -n "$RESUME_FROM" ]]; then
  CMD+=( --resume-from "$RESUME_FROM" )
fi

# Propagar flags adicionales del usuario
CMD+=( "$@" )

# Ejecutar
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

###############################################################################
# 6) Fin
###############################################################################
deactivate
echo "✅ Entrenamiento (re)iniciado — checkpoints en: $OUT_DIR"
echo "📝 Logs en: $LOG_DIR"
