""" 
prepare data for data pipeline
""" 

import os
import joblib
import itertools
import random
random.seed(315)


def get_idx_from_days2(data, selected_days, day_columns='Day', mode='train', train_sequence_length=2*24*6, predict_sequence_length=2*24*6, strides=1*6, max_lags=1):
    def func(data):
        return data.tail(predict_sequence_length - 1).index.tolist()
    
    def func2(data):
        return data.head(max(train_sequence_length, max_lags) + 1).index.tolist()

    cpu_count = os.cpu_count()
    all_idx = data.loc[data[day_columns].isin(selected_days)].index.tolist()
    data_grouped = data.groupby(['TurbID'])
    dropidx = joblib.Parallel(cpu_count)(joblib.delayed(func)(group) for name, group in data_grouped)
    dropidx = list(itertools.chain(*dropidx))

    if mode == 'train':
        dropidx2 = joblib.Parallel(cpu_count)(joblib.delayed(func2)(group) for name, group in data_grouped)
        dropidx2 = list(itertools.chain(*dropidx2))      
        dropidx += dropidx2   

    idx = sorted(list(set(all_idx) - set(dropidx)))
    return idx
