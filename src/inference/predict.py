# -*-Encoding: utf-8 -*-

"""
Description: A demo of the forecasting method
"""

import os
import gc
import time
import joblib
# from functools import cache
import numpy as np
import pandas as pd
import tensorflow as tf


# ----------------------------NN 1 -----------------------------------

class CFG:
    use_models = ['bert']
    train_sequence_length = 2 * 24 * 6
    predict_sequence_length = 2 * 24 * 6
    target_aggs = 1
    test_batch_size = 134
    scaler = 'feature_scaler.pkl'
    target_scaler = None
    feature_column_norm = ['Wspd', 'Wdir', 'Prtv']
    feature_column_short = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']


def custom_loss(y_true, y_pred):   
    y_true1, mask = tf.split(y_true, 2, axis=-1)   
    mask = tf.cast(mask, dtype=tf.float32)  
    y_true1 *= mask
    y_pred *= mask
    rmse_score = tf.math.sqrt(tf.reduce_mean(tf.square(y_true1 - y_pred)))
    return rmse_score


def build_nn_feature(data):
    feature_cols = []
    data = get_lag_rolling_feature(data, roll_column='Wspd', period=6, lags=[6, 144], id_col='TurbID', agg_funs=['mean', 'max'], feature_cols=feature_cols)
    return data, feature_cols


def build_test_data(settings):
    df_raw = pd.read_csv(settings['path_to_test_x'])  # each prediction file has 14 days data
    df_raw[['Wspd', 'Wdir', 'Prtv', 'Patv']] = df_raw.groupby(['TurbID'])[['Wspd', 'Wdir', 'Prtv', 'Patv']].apply(lambda x: x.ffill().bfill())

    latest_hist = df_raw.groupby(['TurbID'])['Patv'].tail(3).values.reshape(-1, 3)
    latest_hist = np.mean(latest_hist, axis=1)
    latest_hist = np.tile(latest_hist.reshape(-1, 1, 1), (1, 3, 1))
    
    # scaler or not
    if CFG.scaler is not None:
        scaler = joblib.load(os.path.join(settings['checkpoints'], CFG.scaler))
        df_raw[CFG.feature_column_norm] = scaler.transform(df_raw[CFG.feature_column_norm])

    turbine_id = df_raw.groupby(['TurbID'])['TurbID'].tail(1).values.reshape(-1, 1).astype(np.int32)
    df_raw['Hour'] = pd.to_datetime(df_raw['Tmstamp'], format='%H:%M').dt.hour
    df_raw['Minute'] = pd.to_datetime(df_raw['Tmstamp'], format='%H:%M').dt.minute
    df_raw['MinuteofDay'] = df_raw['Hour'] * 6 + df_raw['Minute'] / 10
    df_raw['DayofWeek'] = df_raw['Day'] // 7

    df_raw, feature_cols = build_nn_feature(df_raw)
    feature_idx = [df_raw.columns.get_loc(c) for c in feature_cols]

    history_features = df_raw.groupby(['TurbID']).tail(CFG.train_sequence_length).values[:, list(range(3, 13))+feature_idx]
    history_features = history_features.reshape(134, CFG.train_sequence_length, -1).astype(np.float32)   
  
    encoder_features_long = df_raw[['Hour', 'MinuteofDay', 'DayofWeek']].iloc[-CFG.train_sequence_length:].values
    decoder_features_long = df_raw[['Hour', 'MinuteofDay', 'DayofWeek']].iloc[-CFG.predict_sequence_length:].values
    features_long = np.concatenate([encoder_features_long, decoder_features_long], axis=0)    
    features_long = np.tile(np.expand_dims(features_long, 0), (134, 1, 1))   
    
    dataset = tf.data.Dataset.from_tensor_slices(((turbine_id, history_features, features_long), np.ones((134, 288, 1))))
    dataset = dataset.batch(CFG.test_batch_size)  # .prefetch(tf.data.experimental.AUTOTUNE)

    last = df_raw.iloc[-1]['MinuteofDay']    
    shift = int((last + 1) % 144)
    all_history = pd.read_csv(os.path.join(settings['data_path'], settings['filename']))
    all_history['Hour'] = pd.to_datetime(all_history['Tmstamp'], format='%H:%M').dt.hour
    all_history['Minute'] = pd.to_datetime(all_history['Tmstamp'], format='%H:%M').dt.minute
    all_history['MinuteofDay'] = all_history['Hour'] * 6 + all_history['Minute'] / 10
    
    turb_day = all_history.groupby(['TurbID', 'MinuteofDay'])['Patv'].agg('mean')
    turb_day_unstack = turb_day.unstack(level=-1).values
    turb_day_shift = np.roll(turb_day_unstack, -1 * shift, axis=1)
    turb_day_shift = (turb_day_shift - 205.) / (527. - 205.)

    turb_day_shift = turb_day_shift * 36  # base

    top70_max = np.percentile(turb_day_shift, 33)
    turb_day_shift[np.where(turb_day_shift > top70_max)] = turb_day_shift[
                                                               np.where(turb_day_shift > top70_max)] * 2.25

    top50_max = np.percentile(turb_day_shift, 50)
    turb_day_shift[np.where(turb_day_shift > top50_max)] = turb_day_shift[
                                                               np.where(turb_day_shift > top50_max)] * 1.55

    top10_max = np.percentile(turb_day_shift, 90)
    turb_day_shift[np.where(turb_day_shift > top10_max)] = turb_day_shift[
                                                               np.where(turb_day_shift > top10_max)] * 1.25

    top4_max = np.percentile(turb_day_shift, 96)
    turb_day_shift[np.where(turb_day_shift > top4_max)] = turb_day_shift[
                                                              np.where(turb_day_shift > top4_max)] * 1.15

    turb_day_shift = np.concatenate([turb_day_shift, turb_day_shift], axis=1)
    turb_day_shift = np.expand_dims(turb_day_shift, 2)
    del turb_day, df_raw
    gc.collect()
    
    return dataset, turb_day_shift, latest_hist


# @cache
def load_model(settings):
    models = []
    for m in CFG.use_models:
        models.append(
            tf.keras.models.load_model(os.path.join(settings['checkpoints'], m), custom_objects={'custom_loss': custom_loss})
            )
    return models


def forecast(settings):  # official required
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings: (dict)
    Returns:
        The predictions: np.ndarray
    """
    test_loader, turb_day_shift, latest_hist = build_test_data(settings)
    models = load_model(settings)
    res = []

    for model in models:
        time_res = []
        for x in test_loader:
            y = model(x)  # batch * 288 * 1
            time_res.append(y.numpy())
        time_res = np.concatenate(time_res)  # 134 * 288 * 1
        res.append(time_res)

    res = np.array(res)  # + 5
    res = np.mean(res, axis=0)  # n_turbine * n_time * 1
    res = res + turb_day_shift
    res = np.concatenate([latest_hist, res[:, 3:, :]], axis=1)
    res = np.clip(res, 1, 1520)

    tf.keras.backend.clear_session()
    del test_loader, models
    gc.collect()
    return res  # n_turbine * n_time * 1


def get_lag_rolling_feature(data, roll_column, lags, period, id_col, agg_funs, feature_cols=[]):
    # (lag, lag+period）
    for lag in lags:
        for agg_fun in agg_funs:
            feature_col = roll_column + '_lag{}_roll{}_{}'.format(lag, period, agg_fun)
            feature_cols.append(feature_col)
            if id_col is not None:
                data[feature_col] = data.groupby(id_col)[roll_column].transform(lambda x: x.shift(lag+1).rolling(period).agg(agg_fun))
            else:
                data[feature_col] = data[roll_column].shift(lag+1).rolling(period).agg(agg_fun)
    return data


if __name__ == '__main__':
    from prepare import prep_env
    settings = prep_env()
    settings['path_to_test_x'] = '../data/raw/wtbdata_245days.csv'
    settings['data_path'] = '../data/raw'
    
    t1 = time.time()
    y = forecast(settings)
    print("cost {}s".format(time.time() - t1))
    print(y)
    print(y.shape)
