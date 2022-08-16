#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2022-04
# A general trainer for TensorFlow2


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from copy import copy, deepcopy

__all__ = ['KerasTrainer']


class KerasTrainer(object):
    def __init__(self, build_model, loss_fn, optimizer, lr_scheduler=None, strategy=None):
        """
        model: a tf.keras.Model instance
        loss: a loss function
        optimizer: tf.keras.Optimizer instance
        """
        self.build_model = build_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler      
        self.strategy = strategy  

    def train(
        self, 
        train_dataset, 
        valid_dataset=None, 
        n_epochs=10, 
        batch_size=32, 
        steps_per_epoch=None,        
        callback_eval_metrics=None, 
        transform=None, 
        early_stopping=None, 
        checkpoint=None, 
        verbose=0, 
        **kwargs):
        """
        train_dataset: tf.data.Dataset instance, or [x_train, y_train]
        valid_dataset: None or tf.data.Dataset instance, or [x_valid, y_valid]
        transform2label: transform function from logit to label
        """
        callbacks = []
        if early_stopping is not None:
            callbacks.append(early_stopping)
        if checkpoint is not None:
            callbacks.append(checkpoint)
        if 'callbacks' in kwargs:
            callbacks += kwargs.get('callbacks')
            print('callback', callbacks)
        
        # if self.strategy is None:
        #     self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        # else:
        #     train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        #     if valid_dataset is not None:
        #         valid_dataset = self.strategy.experimental_distribute_dataset(valid_dataset) 
        
        #with self.strategy.scope():
        self.model = self.build_model()
        # print(self.model.summary())
        self.model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=callback_eval_metrics, run_eagerly=True) 
        if isinstance(train_dataset, (list, tuple)):
            x_train, y_train = train_dataset
            x_valid, y_valid = valid_dataset
            self.history = self.model.fit(
                x_train, 
                y_train,
                validation_data=(x_valid, y_valid), 
                steps_per_epoch=steps_per_epoch,
                epochs=n_epochs, 
                batch_size=batch_size, 
                verbose=verbose, 
                callbacks=callbacks)
        else:
            self.history = self.model.fit(
                train_dataset, 
                validation_data=valid_dataset, 
                steps_per_epoch=steps_per_epoch,
                epochs=n_epochs, 
                batch_size=batch_size, 
                verbose=verbose, 
                callbacks=callbacks)
        return self.history

    def predict(self, x_test, method=None, batch_size=1):
        y_test_pred = self.model.predict(x_test, batch_size=batch_size)
        return y_test_pred
    
    def get_model(self):
        return self.model

    def save_model(self, model_dir, only_pb=True, checkpoint_dir=None):
         # save the model
        if checkpoint_dir is not None:
            print('check', checkpoint_dir)
            self.model.load_weights(checkpoint_dir)
        else:
            print('nocheck')

        self.model.save(model_dir)
        print("protobuf model successfully saved in {}".format(model_dir))

        if not only_pb:
            self.model.save_weights("{}.ckpt".format(model_dir))
            print("model weights successfully saved in {}.ckpt".format(model_dir))
        return

