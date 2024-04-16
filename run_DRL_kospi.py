import os
os.environ["CUDA_VISIBLE_DEVICES"] = "/device:GPU:1"

import warnings
warnings.filterwarnings('ignore')

# common library
import pandas as pd
import numpy as np
import tensorflow as tf
from stable_baselines.common.vec_env import DummyVecEnv
# config
from config.config_kospi import *
# preprocessor
from preprocessing.preprocessors_kospi import * # from folder.file import *
# model
from model.models_kospi import *

def run_model() -> None:
    """Train the model."""

    # read and preprocess data

    preprocessed_path = "data/done_data.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0) # 0번째 column을 인덱스로 지정
    else:
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)


    # print(data.head())
    # print(data.size)
    

    # 2015/10/01 is the date that validation starts 
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    # 하지만 코드에 따르면 2015/10/02부터 validation 시작
    
    # trade는 2020년 07월 6일까지 됨
    unique_trade_date = data[(data.datadate > 20151001)&(data.datadate <= 20200707)].datadate.unique()
    #print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63 # default=63
    validation_window = 63 # default=63
    
    
    ## Original Ensemble Strategy
    #run_ensemble_strategy(df=data, 
    #                      unique_trade_date=unique_trade_date, 
    #                      rebalance_window=rebalance_window, 
    #                      validation_window=validation_window)
    
    ## Remake Ensemble
    #run_remake_ensemble(df=data, 
    #              unique_trade_date= unique_trade_date,
    #              rebalance_window = rebalance_window,
    #              validation_window= validation_window)
    
    ## Renewal Ensemble2
    #run_ensemble2(df=data, 
    #              unique_trade_date= unique_trade_date,
    #              rebalance_window = rebalance_window,
    #              validation_window= validation_window)
    
    ## PPO
    #run_ppo(df=data, 
    #        unique_trade_date= unique_trade_date,
    #        rebalance_window = rebalance_window,
    #        validation_window= validation_window)
    
    # A2C
    run_a2c(df=data, 
            unique_trade_date= unique_trade_date,
            rebalance_window = rebalance_window,
            validation_window= validation_window)
    
    ## DDPG
    #run_ddpg(df=data, 
    #        unique_trade_date= unique_trade_date,
    #        rebalance_window = rebalance_window,
    #        validation_window= validation_window)
    
    ## ACKTR
    #run_acktr(df=data, 
    #       unique_trade_date= unique_trade_date,
    #        rebalance_window = rebalance_window,
    #       validation_window= validation_window)
    
    ## TRPO
    #run_trpo(df=data, 
    #        unique_trade_date= unique_trade_date,
    #        rebalance_window = rebalance_window,
    #        validation_window= validation_window)
    
    ## SAC
    #run_sac(df=data, 
    #        unique_trade_date= unique_trade_date,
    #        rebalance_window = rebalance_window,
    #        validation_window= validation_window)
    
    ## TD3
    #run_td3(df=data, 
    #        unique_trade_date= unique_trade_date,
    #        rebalance_window = rebalance_window,
    #        validation_window= validation_window)
    
    ## GAIL
    #run_gail(df=data, 
    #        unique_trade_date= unique_trade_date,
    #        rebalance_window = rebalance_window,
    #        validation_window= validation_window)
    
if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        run_model()