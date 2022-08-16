""" 
prepare the feature
"""

import numpy as np
import pandas as pd


def get_lag_rolling_feature(data, roll_column, lags, period, id_col, agg_funs, feature_cols=[]):
    # stat between (lag, lag+period） 
    for lag in lags:
        for agg_fun in agg_funs:
            feature_col = roll_column + '_lag{}_roll{}_{}'.format(lag, period, agg_fun)
            feature_cols.append(feature_col)
            if id_col is not None:
                data[feature_col] = data.groupby(id_col)[roll_column].transform(lambda x: x.shift(lag+1).rolling(period).agg(agg_fun))
            else:
                data[feature_col] = data[roll_column].shift(lag+1).rolling(period).agg(agg_fun)
    return data

