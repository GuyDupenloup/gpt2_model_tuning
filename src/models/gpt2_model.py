# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class LoRALayer(tf.keras.layers.Layer):
    """
    Low-Rank Adaptation layer.
    Matrices A and B are initialized as described in the original LoRA paper.
    """
    def __init__(self, output_size, rank=8, alpha=16, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.output_size = output_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
    
        # A: projects from input to rank
        self.lora_A = tf.keras.layers.Dense(
            rank,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name='lora_A'
        )
        
        # B: projects from rank to output_size
        self.lora_B = tf.keras.layers.Dense(
            output_size,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Zeros(),
            name='lora_B'
        )
    
    def call(self, inputs):
        return self.lora_B(self.lora_A(inputs)) * self.scaling

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_size': self.output_size,
            'rank': self.rank,
            'alpha': self.alpha
        })
        return config
    

@tf.keras.utils.register_keras_serializable()
class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head self-attention mechanism with optional LoRA adaptation.
    Implements causal (autoregressive) attention with configurable number of heads.
    """
    def __init__(self, seq_len, d_model, n_heads, lora_config=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.seq_len = seq_len
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.lora_config = lora_config
        self.use_lora = lora_config is not None

        # -inf constant giving 0.0 after softmax
        self.epsilon = tf.constant(-1e9, dtype=tf.float32)

        # Causal mask (triangular matrix)
        # Shape (batch, 1, 1, seq_len) to mask out the keys
        self.causal_mask = tf.linalg.band_part(tf.ones((1, 1, seq_len, seq_len), dtype=tf.bool), -1, 0)

        # Concatenated Wq, Wk and Wv matrices
        self.W_qkv = tf.keras.layers.Dense(3 * d_model, name='W_qkv')

        if self.use_lora:
            self.W_qkv_lora = LoRALayer(
                3 * d_model,
                rank=lora_config['rank'],
                alpha=lora_config['alpha'],
                name='W_qkv_lora'
        )

        # Output projection matrix
        self.output_proj = tf.keras.layers.Dense(d_model, name='out_proj')

        if self.use_lora:
            self.c_proj_lora = LoRALayer(
                d_model, 
                rank=lora_config['rank'],
                alpha=lora_config['alpha'],                
                name='c_proj_lora'
        )

    def call(self, input, attention_mask):

        # Get the batch size
        batch = tf.shape(input)[0]

        # Multiply inputs by query/key/value weight matrices
        if self.use_lora:
            # Add LoRA matrices contribution
            QKV = self.W_qkv(input) + self.W_qkv_lora(input)
        else:
            QKV = self.W_qkv(input)

        # Separate Q/K/V
        # Shape: (batch, d_model)
        Q, K, V = tf.split(QKV, num_or_size_splits=3, axis=-1)

        # d_model = d_head * n_heads
        Q = tf.reshape(Q, (batch, self.seq_len, self.n_heads, self.d_head))
        K = tf.reshape(K, (batch, self.seq_len, self.n_heads, self.d_head))
        V = tf.reshape(V, (batch, self.seq_len, self.n_heads, self.d_head))

        # (batch, seq_len, n_heads, d_head) -> (batch, n_heads, seq_len, d_head)
        Q = tf.transpose(Q, perm=(0, 2, 1, 3))
        K = tf.transpose(K, perm=(0, 2, 1, 3))
        V = tf.transpose(V, perm=(0, 2, 1, 3))

        # Calculate dot products between queries and keys
        # Shape: (batch, n_heads, seq_len, seq_len)
        scores = tf.matmul(Q, tf.transpose(K, perm=[0, 1, 3, 2]))

        # Apply attention mask to mask out padding tokens in keys
        scores = tf.where(attention_mask[:, None, None, :] == 0, self.epsilon, scores)

        # Apply causal attention mask
        scores = tf.where(self.causal_mask, scores, self.epsilon)

        # Scale the scores and apply softmax to get the attention weights
        scaled_scores = scores / tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        attn_weights = tf.nn.softmax(scaled_scores, axis=-1)

        # Multiply the scaled attention scores by value weight matrix
        context = tf.matmul(attn_weights, V)

        # (batch, n_heads, seq_len, d_head) -> (batch, seq_len, n_heads, d_head)
        context = tf.transpose(context, perm=[0, 2, 1, 3])

        # Reshape to output shape
        context = tf.reshape(context, (batch, self.seq_len, self.d_model))

        # Output projection
        if self.use_lora:
            # Add LoRA matrices contribution
            d_out = self.output_proj(context) + self.c_proj_lora(context)
        else:
            d_out = self.output_proj(context)

        return d_out

    def get_config(self):
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'lora_config': self.lora_config
        })
        return config
    

@tf.keras.utils.register_keras_serializable()
class GPT2FeedForwardNetwork(tf.keras.layers.Layer):
    """
    Position-wise feed-forward network.
    Applies two linear transformations with GELU activation.
    """
    def __init__(self, d_model, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.d_model = d_model

        self.ff_inner = tf.keras.layers.Dense(
            4 * d_model,
            activation=tf.keras.activations.gelu,
            name='ffn_inner',
        )
        self.ff_out = tf.keras.layers.Dense(d_model, name='ffn_out')

    def call(self, input):
        x = self.ff_inner(input)
        x = self.ff_out(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model
        })
        return config
    

@tf.keras.utils.register_keras_serializable()
class GPT2Transformer(tf.keras.layers.Layer):
    """
    GPT2 transformer block.
    Consists of multi-head attention and feed-forward network,
    each with layer normalization and residual connections.
    """
    def __init__(
            self, seq_len, d_model, n_heads, lora_config=None, dropout_rate=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.lora_config = lora_config
        self.dropout_rate = dropout_rate

        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='ln_1')
        self.attn_heads = MultiHeadAttention(seq_len, d_model, n_heads, lora_config=lora_config, name='attn_heads')
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate, name='drop_1')
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='ln_2')
        self.ffn = GPT2FeedForwardNetwork(d_model, name='ffn')
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate, name='drop_2')


    def call(self, input, attention_mask, training=None):

        # First sub-layer
        x1 = self.layer_norm_1(input, training=training)
        x1 = self.attn_heads(x1, attention_mask=attention_mask)
        x1 = self.dropout_1(x1, training=training)

        # First residual connection
        x2 = x1 + input

        # Second sub-layer
        x3 = self.layer_norm_2(x2, training=training)
        x3 = self.ffn(x3, training=training)
        x3 = self.dropout_2(x3, training=training)

        # Second residual connection
        output = x2 + x3

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'lora_config': self.lora_config,
            'dropout_rate': self.dropout_rate
        })
        return config
    

@tf.keras.utils.register_keras_serializable()
class GPT2Model(tf.keras.models.Model):
    """
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

            dropout_rate:
                Dropout rate for dropout layers.
                Applied after embeddings and after each transformer sub-layer.
                Optional, defaults to 0.1

        Model call() method:
        -------------------
            Arguments:
                inputs:
                    Input token IDs.
                    A tf.Tensor with shape (batch_size, seq_len) and data type tf.int32
                attention_mask:
                    Attention mask (0: ignore token, 1: consider token)
                    A tf.Tensor with shape (batch_size, seq_len) and data type tf.int32
            Returns:
                Hidden state logits over vocabulary
                A tf.Tensor of with shape (batch_size, seq_len, vocab_size) and data_type tf.float32
    """

    def __init__(self, model_config, lora_config=None, dropout_rate=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.config = model_config
        self.lora_config = lora_config
        self.dropout_rate = dropout_rate

        # Get model config parameters
        vocab_size, seq_len, d_model, n_layers, n_heads = (
            model_config[k] for k in ('vocab_size', 'seq_len', 'd_model', 'n_layers', 'n_heads')
        )

        # Token and position embedding layers
        self.token_embed_layer = tf.keras.layers.Embedding(vocab_size, d_model, name='tkn_emb')
        self.position_embed_layer = tf.keras.layers.Embedding(seq_len, d_model, name='pos_emb')

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        # Transformer layers
        self.transformer_layers = [
            GPT2Transformer(
                seq_len,
                d_model,
                n_heads,
                lora_config=lora_config,
                dropout_rate=dropout_rate,
                name=f'transformer_{i}'
            ) 
            for i in range(n_layers)
        ]

        self.layer_norm_final = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_f')

        # Token indices for position embeddings
        self.positions = tf.range(start=0, limit=seq_len, delta=1)


    def call(self, inputs, attention_mask, training=None):
        """
        Forward pass through the GPT-2 model.
        """

        # Embeddings
        token_embed = self.token_embed_layer(inputs)
        position_embed = self.position_embed_layer(self.positions)
        x = token_embed + position_embed[None, :, :]

        x = self.dropout(x, training=training)
        
        for transformer in self.transformer_layers:
            x = transformer(x, attention_mask, training=training)

        output = self.layer_norm_final(x, training=training)

        return output


    def freeze_all_but_lora(self):
        """
        Freezes all layers except LoRA matrices
        Must be called before tuning training
        """
        # Check that a LoRA config was provided
        assert self.lora_config is not None

        # Make all the layers trainable
        self.trainable = True
        
        # Freeze embeddings
        self.token_embed_layer.trainable = False
        self.position_embed_layer.trainable = False
        
        # Freeze final layer norm
        self.layer_norm_final.trainable = False
        
        # Process each transformer block
        for transformer in self.transformer_layers:
            # Freeze LayerNorms
            transformer.layer_norm_1.trainable = False
            transformer.layer_norm_2.trainable = False
            
            # Freeze FFN
            transformer.ffn.trainable = False
            
            # Freeze multi-head attention except LoRA matrices
            attn = transformer.attn_heads
            attn.W_qkv.trainable = False
            attn.output_proj.trainable = False

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_config': self.model_config,
            'lora_config': self.lora_config,
            'dropout_rate': self.dropout_rate
        })
        return config
