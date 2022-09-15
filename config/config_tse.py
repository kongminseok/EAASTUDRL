import pathlib

import pandas as pd
import datetime
import os

# data
TRAINING_DATA_FILE = "data/tse_30_2009_2020.csv"

now = datetime.datetime.now()

data = 'tse'
agent = 'a2c_HMAX_NORMALIZE10_hold0to3'
number = 5

TRAINED_MODEL_DIR = f"trained_models/trained_model_{data}_{agent}_{number}"
result = f"result_{data}_{agent}_{number}"
RESULT = f"results/{result}"

try:
    os.makedirs(TRAINED_MODEL_DIR)
    os.makedirs(RESULT)
except:
    pass