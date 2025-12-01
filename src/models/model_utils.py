# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tabulate
import numpy as np
from transformers import TFGPT2LMHeadModel


def get_gpt2_model_config(model_size):

    model_configs = {
         '124M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 768,  'n_layers': 12, 'n_heads': 12},
         '355M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1024, 'n_layers': 24, 'n_heads': 16},
         '774M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1280, 'n_layers': 36, 'n_heads': 20},
        '1542M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1600, 'n_layers': 48, 'n_heads': 25}
    }
    assert model_size in model_configs
    config = model_configs[model_size]
    config['size'] = model_size

    return config


def get_pretrained_variables(model_size):

    mapping = {'124M': 'gpt2', '355M': 'gpt2-medium', '774M': 'gpt2-large', '1542M': 'gpt2-xl'}
    assert model_size in mapping
    hf_name = mapping[model_size]

    model = TFGPT2LMHeadModel.from_pretrained(hf_name, from_pt=True)

    return model.trainable_variables


def print_trainable_variables(model):
    """
    Prints the trainable variables of a model (names, shapes, number of parameters)
    """

    print('\n' + '=' * 80)
    print(f"  Trainable variables of model `{model.config['size']}`")
    print('=' * 80 + '\n')

    headers = ['Variable', 'Shape', '#Params']
    data = []
    total_params = 0

    for var in model.trainable_variables:
        num_params = int(np.prod(var.shape))
        total_params += num_params
        data.append([var.name, var.shape, f'{num_params:,.0f}'])

    print(tabulate(data, headers=headers, tablefmt='pipe', colalign=('left', 'center', 'right')))
    print(f'\nTotal trainable parameters: {total_params:,.0f}')
