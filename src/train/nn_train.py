""" 
train the time series model
"""

import sys
sys.path.insert(0, '../Time-series-prediction')
sys.path.insert(1, '../utils')
sys.path.insert(2, './configs')
sys.path.insert(3, './dataset')

import argparse
import warnings
import functools
import importlib
from copy import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import joblib

from tfts import build_tfts_model, KerasTrainer
from read_data import DataReader, DataLoader
from prepare_data import get_idx_from_days2
from util import set_seed, compress_submitted_zip
from feature import *
warnings.filterwarnings("ignore")
np.random.seed(315)

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


# ----------------------------config-----------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", default='bert_v1', help="config filename")
    parser.add_argument("--seed", type=int, default=3150, required=False, help='seed')
    parser.add_argument("--debug", type=lambda x: (str(x).lower() == 'true'), default=False, required=False, help='debug or not')
    return parser.parse_args()


# ----------------------------data-----------------------------------

def build_data(cfg):
    """ 
    prepare data pipe for model
    """
    data = pd.read_csv(cfg.base_dir + 'raw/wtbdata_245days.csv')    
    if args.debug:
        data = data.loc[data['TurbID'].isin([100])]
           
    print(data.shape, np.max(data['Day']))
    data['start_time'] = data['Day'].astype(str) + ' ' + data['Tmstamp'].astype(str)  # Flag of the first of 288
    valid_df_raw = data.loc[data['Day'].isin(cfg.valid_days)]  # for metrics
    valid_df_raw.to_pickle('../../data/user_data/valid_df_raw.pkl')
 
    data[['Wspd', 'Wdir', 'Patv']] = data.groupby(['TurbID'])[['Wspd', 'Wdir', 'Patv']].apply(lambda x: x.ffill().bfill())
    data['Hour'] = pd.to_datetime(data['Tmstamp'], format='%H:%M').dt.hour
    data['Minute'] = pd.to_datetime(data['Tmstamp'], format='%H:%M').dt.minute  
    data['MinuteofDay'] = data['Hour'] * 6 + data['Minute'] / 10   
    data['DayofWeek'] = data['Day'] // 7
   
    data['is_valid'] = 1  # 1 means valid

    if cfg.scaler:
        data[['Wspd', 'Wdir', 'Prtv']] = cfg.scaler.fit_transform(data[['Wspd', 'Wdir', 'Prtv']])
        joblib.dump(cfg.scaler, '../../submit/checkpoints/feature_scaler.pkl')
    if cfg.target_scaler:
        data[['Patv']] = cfg.target_scaler.fit_transform(data[['Patv']])
        joblib.dump(cfg.target_scaler, '../../submit/checkpoints/target_scaler.pkl')
    
    feature_cols = []
    data = get_lag_rolling_feature(data, roll_column='Wspd', period=6, lags=[6, 144], id_col='TurbID', agg_funs=['mean', 'max'], feature_cols=feature_cols)
    cfg.feature_column_short += feature_cols

    data = data.reset_index(drop=True)  # if sampling, update the index
    target_column_idx = [data.columns.get_loc(c) for c in cfg.target_column]
    feature_column_idx_short = [data.columns.get_loc(c) for c in cfg.feature_column_short]
    feature_column_idx_long = [data.columns.get_loc(c) for c in cfg.feature_column_long]
   
    train_idx = get_idx_from_days2(
        data, 
        cfg.train_days, 
        day_columns='Day', 
        mode='train', 
        train_sequence_length=cfg.train_sequence_length, 
        predict_sequence_length=cfg.predict_sequence_length, 
        strides=cfg.train_strides, max_lags=288)

    train_data_reader = DataReader(
        data, 
        cfg.train_sequence_length, 
        cfg.predict_sequence_length, 
        idx=train_idx, 
        target_aggs=cfg.target_aggs, 
        target_column_idx=target_column_idx, 
        feature_column_idx_short=feature_column_idx_short, 
        feature_column_idx_long=feature_column_idx_long)

    train_data_loader = DataLoader(
        train_data_reader, 
        len(cfg.feature_column_short), 
        len(cfg.feature_column_long))(batch_size=cfg.fit_params['batch_size'], shuffle=True, drop_remainder=True)

    valid_idx = get_idx_from_days2(
        data, 
        cfg.valid_days, 
        day_columns='Day', 
        mode='valid', 
        train_sequence_length=cfg.train_sequence_length, 
        predict_sequence_length=cfg.predict_sequence_length, 
        strides=cfg.valid_strides)

    valid_data_reader = DataReader(
        data, 
        cfg.train_sequence_length, 
        cfg.predict_sequence_length, 
        idx=valid_idx, 
        target_aggs=cfg.target_aggs, 
        target_column_idx=target_column_idx, 
        feature_column_idx_short=feature_column_idx_short, 
        feature_column_idx_long=feature_column_idx_long)

    valid_data_loader = DataLoader(
        valid_data_reader, 
        len(cfg.feature_column_short), 
        len(cfg.feature_column_long))(batch_size=cfg.fit_params['batch_size'] * 2, shuffle=False)
    
    valid_df = data.iloc[valid_idx]
    return train_data_reader, train_data_loader, valid_data_reader, valid_data_loader, valid_df


# ----------------------------model-----------------------------------

class KDD(object):
    def __init__(self, train_sequence_length, predict_sequence_length) -> None:   
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length

    def __call__(self, x, **kwargs):
        _, raw, raw_long  = x  # feature is here
        raw, manual = tf.split(raw, [10, tf.shape(raw)[-1]-10], axis=-1)
        wind_speed, wind_dir, _, _, _, _, _, _, _, active_power = tf.split(raw, 10, axis=-1)
        manual = tf.where(tf.math.is_nan(manual), tf.zeros_like(manual), manual)

        day_of_week, hour_feature, minute_of_day = tf.split(raw_long, 3, axis=-1)

        hour_feature = hour_feature / 23 - 0.5
        minute_of_day = minute_of_day / 143 - 0.5
        day_of_week = day_of_week / 6 - 0.5

        _, decoder_hour_feature = tf.split(hour_feature, [self.train_sequence_length, self.predict_sequence_length], axis=1)       
        _, decoder_minute_feature = tf.split(minute_of_day, [self.train_sequence_length, self.predict_sequence_length], axis=1)
        _, decoder_day_feature = tf.split(day_of_week, [self.train_sequence_length, self.predict_sequence_length], axis=1)

        encoder_features = tf.concat([wind_speed, wind_dir], axis=-1)        
        decoder_features = tf.concat([decoder_hour_feature, decoder_minute_feature, decoder_day_feature], axis=-1)
        decoder_features = tf.cast(decoder_features, tf.float32)
        print('Feature shape', encoder_features.shape, decoder_features.shape)
        return active_power, encoder_features, decoder_features


def build_model(use_model, train_sequence_length, predict_sequence_length=288, target_aggs=1, short_feature_nums=10, long_feature_nums=1):
    inputs = (
        Input([1]),
        Input([train_sequence_length, short_feature_nums]),  # raw feature numbers
        Input([train_sequence_length+predict_sequence_length, long_feature_nums])  # long feature
        )
    teacher_inputs = Input([predict_sequence_length//target_aggs, 1])

    ts_inputs = KDD(train_sequence_length, predict_sequence_length)(inputs)
    outputs = build_tfts_model(
        use_model=use_model, 
        predict_sequence_length=predict_sequence_length//target_aggs, 
        custom_model_params=cfg.custom_model_params)(ts_inputs, teacher_inputs)

    model = tf.keras.Model(inputs={'inputs':inputs, 'teacher': teacher_inputs}, outputs=outputs)
    return model


def custom_loss(y_true, y_pred):   
    true, mask = tf.split(y_true, 2, axis=-1)   
    mask = tf.cast(mask, dtype=tf.float32)  
    true *= mask
    y_pred *= mask
    rmse_score = tf.math.sqrt(tf.reduce_mean(tf.square(true - y_pred)) + 1e-9)
    return rmse_score


# ----------------------------train-----------------------------------

def run_train(cfg):    
    train_data_reader, train_data_loader, valid_data_reader, valid_data_loader, valid_df = build_data(cfg) 
    print(len(train_data_reader), len(valid_data_reader), len(valid_df))  

    build_model_fn = functools.partial(
        build_model, 
        cfg.use_model, 
        train_sequence_length=cfg.train_sequence_length, 
        predict_sequence_length=cfg.predict_sequence_length, 
        target_aggs=cfg.target_aggs, 
        short_feature_nums=len(cfg.feature_column_short), 
        long_feature_nums=len(cfg.feature_column_long)
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate = cfg.fit_params['learning_rate'])
    loss_fn = custom_loss  

    trainer = KerasTrainer(build_model_fn, loss_fn=loss_fn, optimizer=optimizer, strategy=None)
    trainer.train(train_data_loader, valid_dataset=valid_data_loader, **cfg.fit_params)  
    trainer.save_model(
        model_dir=cfg.model_dir + '/checkpoints/{}_{}'.format(cfg.use_model, args.seed), 
        checkpoint_dir=cfg.checkpoint_dir+'/nn_{}.h5'.format(cfg.use_model)
    )
    compress_submitted_zip(res_dir='../inference', output_dir='../../weights/result')  # save to submit


if __name__ == '__main__':
    args = parse_args()
    cfg = copy(importlib.import_module(args.config).cfg)    
    set_seed(args.seed)

    cfg.fit_params = {
        'n_epochs': 3,
        'batch_size': 1024,
        'learning_rate': 5e-3,
        'verbose': 1,
        'checkpoint': ModelCheckpoint(
            cfg.checkpoint_dir+'/nn_{}.h5'.format(cfg.use_model), 
            monitor='val_loss', 
            save_weights_only=True, 
            save_best_only=False, 
            verbose=1),      
    }
    
    run_train(cfg)
