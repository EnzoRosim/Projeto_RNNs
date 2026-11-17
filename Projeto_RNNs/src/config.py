"""Configurações globais do projeto."""
from pathlib import Path

# ---- Caminhos ----
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "database" / "data"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- Reprodutibilidade ----
SEED = 42

# ---- Dados ----
TARGET = "Global_active_power"
FREQ = "1h"                # reamostragem (original: 1 min)
USE_TIME_FEATURES = True   # hora do dia e dia da semana (seno/cosseno)

# ---- Janela e horizonte ----
WINDOW = 24                # horas de histórico na entrada
HORIZON = 1                # prever t+1 (1 hora à frente)

# ---- Split temporal (o restante vira teste) ----
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# ---- Hiperparâmetros padrão (serão ajustados no tuning) ----
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0
LR = 1e-3
WEIGHT_DECAY = 0.0
EPOCHS = 100
PATIENCE = 10              # early stopping
