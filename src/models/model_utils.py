# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
from tabulate import tabulate
import numpy as np
import tensorflow as tf
from models.gpt2_language_model import GPT2LanguageModel
from models.gpt2_classifier import GPT2Classifier
from transformers import GPT2LMHeadModel, TFGPT2LMHeadModel


def get_gpt2_model_config(model_size):
    """
    Gets model configuration parameters for each of OpenAI's model sizes.
    """
    model_configs = {
         '124M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 768,  'n_layers': 12, 'n_heads': 12},
         '355M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1024, 'n_layers': 24, 'n_heads': 16},
         '774M': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1280, 'n_layers': 36, 'n_heads': 20},
        '1.56B': {'vocab_size': 50257,  'seq_len': 1024, 'd_model': 1600, 'n_layers': 48, 'n_heads': 25}
    }
    assert model_size in model_configs
    config = model_configs[model_size]
    config['size'] = model_size

    return config


def get_pretrained_variables(model_size):
    """
    Creates a Hugging Face pretrained model of a specified size.
    Returns the trainable variables of the model.
    """

    mapping = {'124M': 'gpt2', '355M': 'gpt2-medium', '774M': 'gpt2-large', '1542M': 'gpt2-xl'}
    assert model_size in mapping
    hf_name = mapping[model_size]

    model = TFGPT2LMHeadModel.from_pretrained(hf_name, from_pt=True)

    return model.trainable_variables


def create_gpt2_language_model(model_size, lora_config=None, dropout_rate=0.1, name='gpt2_lm'):
    """
    Creates a GPT-2 language model (base GPT-2 model with a language modelling output layer).
    OpenAI's pretrained weights are loaded in the model.

    Arguments:
        model_size:
            '124M', '355M', '774M', or '1542M'.
        lora_config:
            Optional LoRA configuration, a dictionary.
            If present, keys must include 'rank' and 'alpha'.
            If not present, LoRA layers are inactive.
        dropout_rate:
            Dropout rate for dropout layers.
            Applied after embeddings and after each transformer sub-layer.
            Optional, defaults to 0.1

    Returns:
        A tf.keras.models.Model object.
        Pretrained GPT-2 language model of the specified size.
    """

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
    num_pretrained_vars = len(pretrained_vars)

    if lora_config is not None:
        model.gpt2_model.freeze_all_but_lora()

        # Trainable variables of the model are the 4 LoRA
        # matrices in each transformer block (4 variables)
        assert len(model.trainable_variables) == 4 * model.config['n_layers']

        # We must filter the non-trainable variables that are not
        # part of the model: metrics_trackers, generator_seeds, etc.
        target_vars = []
        for var in model.non_trainable_variables:
            if 'gpt2_model' in var.name:
                target_vars.append(var)
        assert len(target_vars) == num_pretrained_vars
    else:
        target_vars = model.trainable_variables

    # The source/target variable lists must have the same length.
    assert len(target_vars) == num_pretrained_vars

    for i in range(num_pretrained_vars):
        var = target_vars[i]
        weights = var.numpy()

        weights_pt = pretrained_vars[i].numpy()
        # Convert shapes (1, N) to (N,)
        weights_pt = np.squeeze(weights_pt)

		# Check that the weight shapes match and copy weights
        assert weights.shape == weights_pt.shape
        var.assign(weights_pt)

    return model


def create_gpt2_classifier(model_size, num_classes, lora_config=None, dropout_rate=0.1, name='gpt2_classifier'):
    """
    Creates a GPT-2 language model (base GPT-2 model with a classification output layer).
    OpenAI's pretrained weights are loaded in the model.

    Arguments:
        model_size:
            '124M', '355M', '774M', or '1542M'.
        num_classes:
            Number of classes.
        lora_config:
            Optional LoRA configuration, a dictionary.
            If present, keys must include 'rank' and 'alpha'.
            If not present, LoRA layers are inactive.
        dropout_rate:
            Dropout rate for dropout layers.
            Applied after embeddings and after each transformer sub-layer.
            Optional, defaults to 0.1
            
    Returns:
        A tf.keras.models.Model object.
        Pretrained GPT-2 classification model of the specified size.
    """

    model_config = get_gpt2_model_config(model_size)
    model = GPT2Classifier(
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
    num_pretrained_vars = len(pretrained_vars)

    if lora_config is not None:
        model.gpt2_model.freeze_all_but_lora()

        # Trainable variables of the model include:
        #   4 variables for the LoRA matrices in each transformer block
        #   2 variables for the classifier.
        assert len(model.trainable_variables) == 4 * model.config['n_layers'] + 2

        # Filter non-trainable variables that don't belong
        # to the model: metrics trackers, RNG state, etc.
        target_vars = []
        for var in model.non_trainable_variables:
            if 'gpt2_model' in var.name:
                target_vars.append(var)
        assert len(target_vars) == num_pretrained_vars
    else:
        target_vars = model.trainable_variables
        assert len(target_vars) == num_pretrained_vars + 2
    
    for i in range(num_pretrained_vars):
        var = target_vars[i]
        weights = var.numpy()

        weights_pt = pretrained_vars[i].numpy()
        # Convert shapes (1, N) to (N,)
        weights_pt = np.squeeze(weights_pt)

		# Check that the weight shapes match and copy weights
        assert weights.shape == weights_pt.shape
        var.assign(weights_pt)

    return model


def print_model_variables(model, trainable=True, non_trainable=False):
    """
    Prints the trainable/non-trainable variables of a model
    (names, shapes, number of parameters)
    """

    def print_vars(model_size, var_list, var_type):
        print('\n' + '=' * 80)
        print(f"  {var_type} variables of model `{model_size}`")
        print('=' * 80 + '\n')

        total_params = 0
        if len(var_list) > 0:
            data = []
            total_params = 0
            for var in var_list:
                num_params = int(np.prod(var.shape))
                total_params += num_params
                data.append([f'{var.name}', f'{var.shape}', f'{num_params:,.0f}'])

            headers = ['Variable', 'Shape', '#Params']
            print(tabulate(data, headers=headers, tablefmt='pipe', colalign=('left', 'center', 'right')))
        print(f'\nTotal {var_type} parameters: {total_params:,.0f}')

    model_size = model.config['size']
    if trainable:
        print_vars(model_size, model.trainable_variables, 'Trainable')

    if non_trainable:
        print_vars(model_size, model.non_trainable_variables, 'Non-trainable')
