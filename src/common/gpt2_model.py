# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf


class LoRALayer(tf.keras.layers.Layer):
    """Low-Rank Adaptation layer that wraps a Dense layer."""
    def __init__(self, output_size, rank=8, alpha=16, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

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


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, lora_config=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.use_lora = lora_config is not None


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
        self.c_proj = tf.keras.layers.Dense(d_model, name='c_proj')

        if self.use_lora:
            self.c_proj_lora = LoRALayer(
                d_model, 
                rank=lora_config['rank'],
                alpha=lora_config['alpha'],                
                name='c_proj_lora'
        )

    def call(self, input, attention_mask=None, training=False):

        batch, seq_len, _= tf.unstack(tf.shape(input))

        # Get queries, keys and values
        if self.use_lora:
            QKV = self.W_qkv(input) + self.W_qkv_lora(input)
        else:
            QKV = self.W_qkv(input)

        Q, K, V = tf.split(QKV, num_or_size_splits=3, axis=-1)

        # d_model = d_head * n_heads
        Q = tf.reshape(Q, (batch, seq_len, self.n_heads, self.d_head))
        K = tf.reshape(K, (batch, seq_len, self.n_heads, self.d_head))
        V = tf.reshape(V, (batch, seq_len, self.n_heads, self.d_head))

        # Transpose from (batch, seq_len, n_heads, d_head)
        # to (batch, n_heads, seq_len, d_head)
        Q = tf.transpose(Q, perm=(0, 2, 1, 3))
        K = tf.transpose(K, perm=(0, 2, 1, 3))
        V = tf.transpose(V, perm=(0, 2, 1, 3))

        # Calculate the attention scores (dot products between queries and keys)
        # Shape: (batch, seq_len, d_model)
        scores = tf.matmul(Q, tf.transpose(K, perm=[0, 1, 3, 2]))

        epsilon = tf.constant(-1e9, dtype=tf.float32)
        if attention_mask is not None:
            # Apply the attention mask (mask out padding tokens in keys)
            attn_mask = attention_mask[:, None, None, :]  # Broadcast to (batch, 1, 1, seq_len) to mask keys
            scores = tf.where(attn_mask == 0, epsilon, scores)

        # Apply causal attention using a triangular matrix
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0)
        scores = tf.where(causal_mask[None, None, :, :], scores, epsilon)

        # Scale the scores and apply softmax to get the attention weights
        scaled_scores = scores / tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        attn_weights = tf.nn.softmax(scaled_scores, axis=-1)

        # Calculate the context vectors
        # shape: (batch, n_heads, seq_len, d_head)
        context = tf.matmul(attn_weights, V)

        # Transpose to have (batch, seq_len, n_heads, d_head)
        context = tf.transpose(context, perm=[0, 2, 1, 3])

        # Reshape to output size
        context = tf.reshape(context, (batch, seq_len, self.d_model))

        # Output projection layer
        if self.use_lora:
            d_out = self.c_proj(context) + self.c_proj_lora(context)
        else:
            d_out = self.c_proj(context)

        return d_out


class GPT2FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.ff_inner = tf.keras.layers.Dense(
            4 * d_model,
            activation=tf.keras.activations.gelu,
            name='ffn_inner',
        )
        self.ff_out = tf.keras.layers.Dense(d_model, name='ffn_out')

    def call(self, input, training=False):
        x = self.ff_inner(input)
        x = self.ff_out(x)
        return x


class GPT2Transformer(tf.keras.layers.Layer):
    def __init__(
            self, d_model, n_heads, lora_config=None, dropout_rate=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        # 1st LayerNorm layer
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_1')
        
        # Multi-head attention block
        self.attention = MultiHeadAttention(d_model, n_heads, lora_config=lora_config, name='attention')
        
        # 2nd LayerNorm layer
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_2')

        # Feedforward network
        self.ffn = GPT2FeedForwardNetwork(d_model, name='ffn')

        # Dropout layers
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)


    def call(self, input, attention_mask, training=False):

        input_norm = self.norm_1(input)
        attn_out = self.attention(input_norm, attention_mask=attention_mask, training=training)
        attn_out = self.dropout_1(attn_out, training=training)

        # Residual connection
        x = attn_out + input

        x_norm = self.norm_2(x)
        ff_out = self.ffn(x_norm, training=training)
        ff_out = self.dropout_2(ff_out, training=training)

        # Residual connection
        output = x + ff_out

        return output


class GPT2BaseModel(tf.keras.models.Model):
    """
        Arguments:
            dropout_rate:
                Dropout rate for dropout layers (defaults to 0.1)

        Returns:
            Logits over vocabulary
            A tensor of floats with shape (batch_size, seq_len, vocabulary)
    """

    def __init__(self, model_config, lora_config=None, dropout_rate=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.config = model_config
        if lora_config is not None:
            self.lora_config = lora_config

        # Get model config parameters
        vocab_size, seq_len, d_model, n_layers, n_heads = (
            model_config[k] for k in ('vocab_size', 'seq_len', 'd_model', 'n_layers', 'n_heads')
        )

        self.token_embed_layer = tf.keras.layers.Embedding(vocab_size, d_model, name='token_embd')
        self.position_embed_layer = tf.keras.layers.Embedding(seq_len, d_model, name='position_embd')

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        
        self.transformer_blocks = [
            GPT2Transformer(
                d_model,
                n_heads,
                lora_config=lora_config,
                dropout_rate=dropout_rate,
                name=f'transformer_{i}'
            ) 
            for i in range(n_layers)
        ]

        self.norm_f = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='lnorm_f')


    def freeze_all_but_lora(self):

        """
        Freeze all layers except LoRA matrices.
        Useful for LoRA fine-tuning with different task-specific heads.
        """

        assert self.use_lora == True

        self.trainable = True
        
        # Freeze embeddings
        self.token_embed_layer.trainable = False
        self.position_embed_layer.trainable = False
        
        # Freeze final layer norm
        self.norm_f.trainable = False
        
        # Process each transformer block
        for transformer_layer in self.transformer_blocks:
            # Freeze LayerNorms
            transformer_layer.norm_1.trainable = False
            transformer_layer.norm_2.trainable = False
            
            # Freeze FFN
            transformer_layer.ffn.trainable = False
            
            # Freeze multi-head attention except LoRA matrices
            attention_layer = transformer_layer.attention
            attention_layer.W_qkv.trainable = False
            attention_layer.c_proj.trainable = False


    def call(self, data, training=False):

        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        
        token_embed = self.token_embed_layer(input_ids)
        self.embedding_weights = token_embed

        # Add position embeddings
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        position_embed = self.position_embed_layer(positions)  # Shape: (seq_len, d_model)
        x = token_embed + position_embed[None, :, :]

        x = self.dropout(x, training=training)
        
        for block in self.transformer_blocks:
            x = block(x, attention_mask, training=training)

        output = self.norm_f(x)

        return output
