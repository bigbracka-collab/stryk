import os

# Projektrot (den mapp där config.py ligger)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Sökvägar relativt till projektroten
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELL_DIR = os.path.join(ROOT_DIR, "modeller")
PARAM_DIR = os.path.join(ROOT_DIR, "modellparametrar")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# Standardinställningar för modellträning och prognos
DEFAULT_FORM_FACTOR = 1.0
DEFAULT_ZERO_INFLATION = False
DEFAULT_MAX_GOALS = 5
