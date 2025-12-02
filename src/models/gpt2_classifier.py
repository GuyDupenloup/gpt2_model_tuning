# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
from models.gpt2_model import GPT2Model


class GPT2Classifier(tf.keras.models.Model):
    """
    Implements OpenAI's GPT-2 model with a classification head.


    Model instantiation arguments:
    -----------------------------
        model_config:
            The model configuration, a dictionary.
            Keys must include:
                'vocab_size': vocabulary size
                'seq_len': input sequence length (context size)
                'd_model': hidden state size (embeddings size)
                'n_layers': number of transformer blocks
                'n_heads': number of attention heads

        lora_config:
            Optional LoRA configuration, a dictionary.
            If present, keys must include 'rank' and 'alpha'.
            If not present, LoRA layers are inactive.

        num_classes: number of classes.

        dropout_rate:
            Dropout rate for dropout layers.
            Applied after embeddings and after each transformer sub-layer.
            Optional, defaults to 0.1

    Model call() method:
    -------------------
        Arguments:
            inputs:
                A dictionary with the following items:
                    'input_ids':
                        Token IDs of the input sequence.
                        A tf.tensor with shape (batch_size, seq_len) and data type tf.int32
                    'attention mask':
                        Attention mask to remove padding tokens from consideration.
                        0: ignored, 1: considered
                        A tf.tensor with shape (batch_size, seq_len) and data type tf.int32

        Returns:
            Hidden state logits over vocabulary
            A tf.Tensor of with shape (batch_size, seq_len, vocab_size) and data_type tf.float32
    """

    def __init__(
        self, model_config, num_classes, lora_config=None, pooling='last', dropout_rate=0.1, name=None, **kwargs):
	
        super().__init__(name=name, **kwargs)
        self.config = model_config
        self.lora_config = lora_config
        self.num_classes = num_classes
        self.pooling = pooling  # 'last' or 'mean'
        
        self.gpt2_model = GPT2Model(model_config, lora_config=lora_config, dropout_rate=dropout_rate, name='gpt2_model')
		
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.classifier = tf.keras.layers.Dense(num_classes, name='classifier')


    def call(self, inputs, training=False):
	
        # Get outputs from GPT-2
        hidden_states = self.gpt2_model(
             inputs['input_ids'],
             inputs['attention_mask'],
             training=training
        )

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
