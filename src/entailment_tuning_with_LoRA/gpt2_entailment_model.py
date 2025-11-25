# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
from common.gpt2_model import GPT2Model


class GPT2EntailmentModel(tf.keras.models.Model):

    def __init__(self, model_config, pooling='last', dropout_rate=0.1, lora_rank=8, lora_alpha=16, name=None, **kwargs):
	
        super().__init__()
        self.config = model_config
        self.pooling = pooling  # 'last' or 'mean'

        self.gpt2_model = GPT2Model(model_config, dropout_rate=0.1, lora_rank=lora_rank, lora_alpha=lora_alpha)
		
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        # This is an entailment model, so there are 2 classes.
        self.classifier = tf.keras.layers.Dense(2, name='classifier')


    def call(self, inputs, training=False):
	
        # Get outputs from GPT-2
        hidden_states = self.gpt2_model(inputs, training=training)

        # Pooling strategy
        if self.pooling == 'last':
            # Use last token
            pooled_output = hidden_states[:, -1, :]
			
        elif self.pooling == 'mean':
            # Mean pooling over valid tokens
            attention_mask = inputs['attention_mask']
            mask = tf.cast(tf.expand_dims(attention_mask, axis=-1), tf.float32)
            pooled_output = tf.reduce_sum(hidden_states * mask, axis=1) / tf.reduce_sum(mask, axis=1)

        # Classification head
        x = self.dropout(pooled_output, training=training)
        logits = self.classifier(x)
        return logits
