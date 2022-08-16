""" 
Bert encoder
"""

from default_config import basic_cfg


cfg = basic_cfg
cfg.use_model = 'bert'

cfg.custom_model_params = {
    'n_encoder_layers': 1,
    'use_token_embedding': False,
    'attention_hidden_sizes': 32*1,
    'num_heads': 1,
    'attention_dropout': 0.,
    'ffn_hidden_sizes': 32,
    'ffn_filter_sizes': 32,  # should be same with attention_hidden_sizes
    'ffn_dropout': 0.,
    'layer_postprocess_dropout': 0.,
    'skip_connect': False
}
