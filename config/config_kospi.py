from pathlib import Path

import pandas as pd
import datetime
import os

# data
TRAINING_DATA_FILE = "data/kospi_30_2009_2020.csv"

now = datetime.datetime.now()

data = 'kospi'
agent = 'a2c_HMAX_NORMALIZE10_hold0to3'
number = 1

TRAINED_MODEL_DIR = f"trained_models/trained_model_{data}_{agent}_{number}"
#result = f"result_{data}_{agent}_{number}"
RESULT = f"results/{data}/{agent}_{number}"

Path(TRAINED_MODEL_DIR).mkdir(exist_ok=True, parents=True)
Path(RESULT).mkdir(exist_ok=True, parents=True)