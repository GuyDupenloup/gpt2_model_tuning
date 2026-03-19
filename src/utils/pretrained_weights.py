# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import numpy as np
from transformers import TFGPT2LMHeadModel
from utils.model_utils import create_gpt2_language_model
from transformers import TFGPT2LMHeadModel


def save_gpt2_pretrained_weights(model_size, filepath):

    model = create_gpt2_language_model(model_size)

    mapping = {'124M': 'gpt2', '355M': 'gpt2-medium', '774M': 'gpt2-large', '1542M': 'gpt2-xl'}
    assert model_size in mapping
    hf_name = mapping[model_size]

    hf_model = TFGPT2LMHeadModel.from_pretrained(hf_name, from_pt=True)

    assert len(model.trainable_variables) == len(hf_model.trainable_variables)

    var_weights = {}
    for i in range(len(model.trainable_variables)):
        var = model.trainable_variables[i]
        hf_var = hf_model.trainable_variables[i]

        # Use var.path if present (var.name could be just a leaf name)
        var_name = var.path if hasattr(var, 'path') else var.name

        weights = hf_var.numpy()
        var_weights[var_name] = np.squeeze(weights)

    np.savez(filepath, **var_weights)
