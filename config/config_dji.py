from pathlib import Path

import pandas as pd
import datetime
import os

# data
#TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
TRAINING_DATA_FILE = "data/dow_30_2009_2020.csv"

data = 'dji'
agent = 'a2c_HMAX_NORMALIZE10_hold0to3'
number = 1

#now = datetime.datetime.now()
#TRAINED_MODEL_DIR = f"trained_models/{now}"
#TRAINED_MODEL_DIR = f"trained_models/{now.date()}"
#TRAINED_MODEL_DIR = f"trained_models/trained_model_{str(now.date())+'-'+str(now.hour)+'-'+str(now.minute)+'-'+str(now.second)}"
TRAINED_MODEL_DIR = f"trained_models/trained_model_{data}_{agent}_{number}"
#result = f"result_{data}_{agent}_{number}"
RESULT = f"results/{data}/{agent}_{number}"

Path(TRAINED_MODEL_DIR).mkdir(exist_ok=True, parents=True)
Path(RESULT).mkdir(exist_ok=True, parents=True)