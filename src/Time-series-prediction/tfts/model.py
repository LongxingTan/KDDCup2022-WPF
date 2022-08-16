#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com

import tensorflow as tf
from .models.bert import Bert


def build_tfts_model(use_model, predict_sequence_length, custom_model_params=None, **kwargs):    
    if use_model.lower() == 'bert':
        Model = Bert(predict_sequence_length=predict_sequence_length, custom_model_params=custom_model_params)    
    else:
        raise ValueError("Unsupported use_model of {}".format(use_model))
    return Model
