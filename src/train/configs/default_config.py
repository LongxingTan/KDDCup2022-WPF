import os
from types import SimpleNamespace
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

cfg = SimpleNamespace(**{})


# paths
cfg.base_dir = '../../data/'
cfg.model_dir = '../inference'
cfg.checkpoint_dir = '../../weights'


# dataset
cfg.train_days = range(1, 181)  # range(1, 231)
cfg.valid_days = range(231, 246)  # range(231, 246)

cfg.train_sequence_length = 2 * 24 * 6  # maximum 14 days, int type
cfg.predict_sequence_length = 2 * 24 * 6
cfg.train_strides = 1  # train sample
cfg.valid_strides = 1  # valid sample
cfg.target_aggs = 1  # target agg size, Todo: some model and loss related update
cfg.lb_sample = 195  # phase 1 online test size


# proprocess
cfg.scaler = MinMaxScaler()
cfg.target_scaler = None 


# model
cfg.use_model = ''
cfg.custom_model_params = None

cfg.target_column = ['Patv', 'is_valid']  # Patv, Wspd
cfg.feature_column_short = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']
cfg.feature_column_long = ['DayofWeek', 'Hour', 'MinuteofDay']


basic_cfg = cfg
