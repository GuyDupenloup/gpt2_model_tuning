# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tabulate
import numpy as np
import tensorflow as tf
from gpt2_language_model import GPT2LanguageModel
from gpt2_classification_model import GPT2ClassificationModel
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


def create_language_model(model_size, lora_config=None, dropout_rate=0.1, name='gpt2_lm'):

    model_config = get_gpt2_model_config(model_size)
    model = GPT2LanguageModel(
        model_config, 
        lora_config=lora_config, 
        dropout_rate=dropout_rate,
        name=name
    )

    # Build the model using dummy inputs
    seq_len = model_config['seq_len']
    vocab_size = model_config['vocab_size']
    dummy_input = {
        'input_ids': tf.random.uniform((1, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32),
        'attention_mask': tf.random.uniform((1, seq_len), minval=0, maxval=2, dtype=tf.int32)
    }
    _ = model(dummy_input)

	# Load Hugging Face model of the same size 
    # get it's trainable variables
    pretrained_vars = get_pretrained_variables(model_size)

    if lora_config is not None:
        model.gpt2_model.freeze_all_but_lora()
        target_vars = model.non_trainable_variables

        # Trainable variables of the model are the 4 LoRA
        # matrices in each transformer block (4 variables)
        assert len(model.trainable_variables) == 4 * model.config['n_layers']

    else:
        target_vars = model.trainable_variables

    # The source/target variable lists must have the same length.
    assert len(pretrained_vars) == len(target_vars) 

    for i in range(len(pretrained_vars)):
        var = target_vars[i]
        weights = var.numpy()

        weights_pt = pretrained_vars[i].numpy()
        # Convert shapes (1, N) to (N,)
        weights_pt = np.squeeze(weights_pt)

		# Check that the weight shapes match and copy weights
        assert weights.shape == weights_pt.shape
        var.assign(weights_pt)

    return model


def create_classification_model(model_size, num_classes, lora_config=None, dropout_rate=0.1, name='gpt2_classifier'):

    model_config = get_gpt2_model_config(model_size)
    model = GPT2ClassificationModel(
        model_config, 
        num_classes,
        lora_config=lora_config,
        dropout_rate=dropout_rate,
        name=name
    )

    # Build the model using dummy inputs
    seq_len = model_config['seq_len']
    vocab_size = model_config['vocab_size']
    dummy_input = {
        'input_ids': tf.random.uniform((1, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32),
        'attention_mask': tf.random.uniform((1, seq_len), minval=0, maxval=2, dtype=tf.int32)
    }
    _ = model(dummy_input)

	# Load Hugging Face model of the same size 
    # get it's trainable variables
    pretrained_vars = get_pretrained_variables(model_size)

    if lora_config is not None:
        model.gpt2_model.freeze_all_but_lora()
        target_vars = model.non_trainable_variables

        # Trainable variables of the model are the 4 LoRA
        # matrices in each transformer block (4 variables)
        assert len(model.trainable_variables) == 4 * model.config['n_layers']

    else:
        target_vars = model.trainable_variables

    # The source/target variable lists must have the same length.
    assert len(pretrained_vars) == len(target_vars) 

    for i in range(len(pretrained_vars)):
        var = target_vars[i]
        weights = var.numpy()

        weights_pt = pretrained_vars[i].numpy()
        # Convert shapes (1, N) to (N,)
        weights_pt = np.squeeze(weights_pt)

		# Check that the weight shapes match and copy weights
        assert weights.shape == weights_pt.shape
        var.assign(weights_pt)

    return model
