import hashlib
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data_preprocess.feature_engineering import FeatureEngineer
from src.data_preprocess.preprocessing import DataPreprocessor
from src.data_preprocess.rnn_data_prep import RNNDataPrep
from src.models import rnn_model
from src.models.rnn_model import train_eval_model
from src.utils.utilities import create_config_dict, get_hash

# Set up logging
logging.basicConfig(level=logging.DEBUG)
