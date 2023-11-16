import socket
from datetime import datetime
from pathlib import Path


# ******************************************************************** #
#                            HIGH LEVEL INFO                           #
# ******************************************************************** #
HOST_NAME = socket.gethostname()
PROJECT_ROOT = Path(__file__).resolve().parent


# -------------------------------------------------------------------- #
def get_current_time():
    return str(datetime.now().strftime("%b%d_%H-%M"))


CURRENT_TIME = get_current_time()
# ******************************************************************** #
#                          DATA & OTHER PATHS                          #
# ******************************************************************** #
RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data/interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data/processed"
RNN_INPUT_DATA_DIR = PROJECT_ROOT / "data/rnn_input"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"
# -------------------------------------------------------------------- #
FF_MW_DATA_FILE = INTERIM_DATA_DIR / "ff-mw.pkl"
