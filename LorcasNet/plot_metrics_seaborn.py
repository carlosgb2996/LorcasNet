# plot_metrics_seaborn.py
#!/usr/bin/env python3
import pandas as pd
import seaborn as sns

# Establecer tema moderno
sns.set_theme(style="whitegrid")

# Ruta al CSV de métricas
csv_path = '/g/data/w09/cg2265/LorcasNet/Pred/metrics.csv'

# Cargar datos
df = pd.read_csv(csv_path)

# Identificar columnas
x_col = df.columns[0]         # Ej. modelo, epoch, etc.
metric_cols = df.columns[1:]  # Cada métrica

# Generar una gráfica por métrica
for metric in metric_cols:
    ax = sns.lineplot(data=df, x=x_col, y=metric, marker='o')
    ax.set_title(f'Métrica: {metric}')
    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(f'{metric}.png')
    fig.clf()
