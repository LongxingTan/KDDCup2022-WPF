#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, SpatialDropout1D, BatchNormalization, AveragePooling1D, GRU
from ..layers.transformer_layer import CustomAttention, SelfAttention, FeedForwardNetwork
from ..layers.embed_layer import DataEmbedding, TokenEmbedding, TokenRnnEmbedding



params = {
    'n_encoder_layers': 1,
    'use_token_embedding': False,
    'attention_hidden_sizes': 32*1,
    'num_heads': 2,
    'attention_dropout': 0.,
    'ffn_hidden_sizes': 32*1,
    'ffn_filter_sizes': 32*1,    
    'ffn_dropout': 0.,
    'layer_postprocess_dropout': 0.,
    'skip_connect': False
}


class Bert(object):
    def __init__(self, predict_sequence_length=3, custom_model_params=None) -> None:
        if custom_model_params:
            params.update(custom_model_params)
        self.params = params        
        self.predict_sequence_length = predict_sequence_length
        print(params)

        self.encoder_embedding = TokenEmbedding(params['attention_hidden_sizes'])
        self.spatial_drop = SpatialDropout1D(0.1)
        self.encoder = Encoder(
            params['n_encoder_layers'], 
            params['attention_hidden_sizes'], 
            params['num_heads'], 
            params['attention_dropout'], 
            params['ffn_hidden_sizes'], 
            params['ffn_filter_sizes'], 
            params['ffn_dropout'])
        
        self.project1 = Dense(predict_sequence_length, activation=None)       
  
        self.drop1 = Dropout(0.25)
        self.dense1 = Dense(512, activation='relu')

        self.drop2 = Dropout(0.25)
        self.dense2 = Dense(1024, activation='relu')  
    
    def __call__(self, inputs, teacher=None):
        # inputs: 
        if isinstance(inputs, (list, tuple)):
            x, encoder_features, decoder_features = inputs
            # encoder_features = tf.concat([x, encoder_features], axis=-1)
        else:  # for single variable prediction
            encoder_features = x = inputs
            decoder_features = None

        encoder_features = self.encoder_embedding(encoder_features)
        memory = self.encoder(encoder_features, src_mask=None)  # batch * train_sequence * (hidden * heads)
        encoder_output = memory[:, -1]
 
        encoder_output = self.drop1(encoder_output)
        encoder_output = self.dense1(encoder_output)
        encoder_output = self.drop2(encoder_output)
        encoder_output = self.dense2(encoder_output)
        encoder_output = self.drop2(encoder_output)

        outputs = self.project1(encoder_output)       
        outputs = tf.expand_dims(outputs, -1)
        return outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_encoder_layers, attention_hidden_sizes, num_heads, attention_dropout, ffn_hidden_sizes, ffn_filter_sizes, ffn_dropout):
        super(Encoder, self).__init__()
        self.n_encoder_layers = n_encoder_layers
        self.attention_hidden_sizes = attention_hidden_sizes
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ffn_hidden_sizes = ffn_hidden_sizes
        self.ffn_filter_sizes = ffn_filter_sizes
        self.ffn_dropout = ffn_dropout
        self.layers = []

    def build(self, input_shape):
        for _ in range(self.n_encoder_layers):
            attention_layer = SelfAttention(self.attention_hidden_sizes,  self.num_heads,  self.attention_dropout)
            feed_forward_layer = FeedForwardNetwork(self.ffn_hidden_sizes, self.ffn_filter_sizes, self.ffn_dropout)
            ln_layer1 = LayerNormalization(epsilon=1e-6, dtype="float32")
            ln_layer2 = LayerNormalization(epsilon=1e-6, dtype="float32")
            self.layers.append([attention_layer, ln_layer1, feed_forward_layer, ln_layer2])
        super(Encoder, self).build(input_shape)    

    def call(self, encoder_inputs, src_mask=None):
        x = encoder_inputs
        for _, layer in enumerate(self.layers):
            attention_layer, ln_layer1, ffn_layer, ln_layer2 = layer
            enc = x
            enc = attention_layer(enc, src_mask)          
            enc1 = ln_layer1(x + enc)  # residual connect
            enc1 = ffn_layer(enc1)
            x = ln_layer2(enc + enc1)
        return x

    def get_config(self):
        config = {
            'n_encoder_layers': self.n_encoder_layers,
            'attention_hidden_sizes': self.attention_hidden_sizes,
            'num_heads': self.num_heads,
            'attention_dropout': self.attention_dropout,
            'ffn_hidden_sizes': self.ffn_hidden_sizes,
            'ffn_filter_sizes': self.ffn_filter_sizes,
            'ffn_dropout': self.ffn_dropout
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
