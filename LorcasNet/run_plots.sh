#!/usr/bin/env bash
# setup_and_run.sh
# 1) Crear y activar un entorno virtual llamado metricsenv
python3 -m venv metricsenv
source metricsenv/bin/activate
module unload python3/3.9.2 pytorch/1.10.0
# 2) Actualizar pip e instalar librerías necesarias
pip install --upgrade pip
pip install pandas seaborn

# 3) Ejecutar el script de generación de gráficas
python3 plot_metrics_seaborn.py

# 4) Desactivar el entorno
deactivate
