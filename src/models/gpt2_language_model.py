# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
from models.gpt2_model import GPT2Model


class GPT2LanguageModel(tf.keras.models.Model):
    """
    Implements OpenAI's GPT-2 model with language modelling head.
    
    The model calculates loss, and metrics including accuracy and perplexity.
    No loss or metrics should be passed in arguments when compiling the model.

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
                A dictionary with the following items:
                    'input_ids':
                        Token IDs of the input sequence.
                        A tf.tensor with shape (batch_size, seq_len) and data type tf.int32
                    'attention mask':
                        Attention mask to remove padding tokens from consideration.
                        0: ignored, 1: considered
                        A tf.tensor with shape (batch_size, seq_len) and data type tf.int32
                    'loss_mask':
                        Loss mask to exclude the instruction and context parts of the prompt
                        from loss calculation.
                        0: excluded, 1: included
                        A tf.tensor with shape (batch_size, seq_len) and data type tf.int32

        Returns:
            Hidden state logits over vocabulary
            A tf.Tensor of with shape (batch_size, seq_len, vocab_size) and data_type tf.float32
    """

    def __init__(self, model_config, lora_config=None, dropout_rate=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.config = model_config
        self.lora_config = lora_config

        self.gpt2_model = GPT2Model(model_config, lora_config=lora_config, dropout_rate=dropout_rate, name=name)

        # Loss and metrics trackers
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.accuracy_tracker = tf.keras.metrics.Mean(name='accuracy')
        self.perplexity_tracker =  tf.keras.metrics.Mean(name='perplexity')


    def call(self, inputs):
        """
        Forward pass through language model.
        """
        gpt2_output = self.gpt2_model(inputs['input_ids'], inputs['attention_mask'])

        # Output linear layer that projects hidden state representations to vocabulary.
        # Weights of the projection matrix are shared with the token embedding matrix.
        embedding_weights = self.gpt2_model.token_embed_layer.embeddings
        logits = tf.matmul(gpt2_output, embedding_weights, transpose_b=True)

        return logits


    def compute_loss(self, input_ids, y_pred, mask):
        """
        Calculates the loss.

        Arguments:
            input_ids: 
                Token IDs of the input sequence.
                Shape: (batch_size, seq_len)
            y_pred:
                Model predictions (logits over the vocabulary).
                Shape: (batch_size, seq_len, vocab_size)
            mask:
                Mask specifying which token positions contribute to the loss.
                Shape: (batch_size, seq_len)
        """

        # Shift inputs to get labels
        y_true = input_ids[:, 1:]

        # Truncate the mask to align with labels
        mask = mask[:, 1:]

        # Drop the last prediction (no next-token label for the final position)
        y_pred = y_pred[:, :-1, :]

        # Calculate cross-entropy loss per token (element wise)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        
        # Apply mask token-wise
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = loss * mask
        
        # Return mean loss over non-masked tokens
        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


    def compute_accuracy(self, input_ids, y_pred, mask):
        """
        Calculates prediction accuracy.

        Arguments:
            input_ids: 
                Token IDs of the input sequence.
                Shape: (batch_size, seq_len)
            y_pred:
                Model predictions (logits over the vocabulary).
                Shape: (batch_size, seq_len, vocab_size)
            mask:
                Mask specifying which token positions should contribute
                to the accuracy calculation (same mask as for the loss).
                Shape: (batch_size, seq_len)
        """

        # Shift inputs to get labels
        y_true = input_ids[:, 1:]

        mask = mask[:, 1:]

        # Drop the last prediction (no next-token label for the final position)
        y_pred = y_pred[:, :-1, :]
        
        # Argmax over the vocabulary dimension -> predicted token IDs
        # Shape: (batch_size, seq_len - 1)
        predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        
        # Compare predictions with true labels (1.0 if correct, 0.0 if wrong)
        correct = tf.cast(tf.equal(predictions, y_true), dtype=tf.float32)
        
        # Apply mask token-wise
        mask = tf.cast(mask, dtype=tf.float32)
        correct_masked = correct * mask
        
        # Calculate accuracy: correct predictions / total non-masked tokens
        accuracy = tf.reduce_sum(correct_masked) / tf.maximum(tf.reduce_sum(mask), 1.0)
        
        return accuracy


    def train_step(self, inputs):
        """
        Performs one training step using next-token prediction.
        Computes the forward pass, loss, gradients, and updates model weights.
        Also updates accuracy and perplexity metrics based on the masked tokens.
        """

        input_ids = inputs['input_ids']
        loss_mask = inputs['loss_mask']

        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.compute_loss(input_ids, y_pred, loss_mask)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [
            g if g is not None else tf.zeros_like(v)
            for g, v in zip(gradients, self.trainable_variables)
        ]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Metrics
        accuracy = self.compute_accuracy(input_ids, y_pred, loss_mask)
        perplexity = tf.exp(loss)

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        self.perplexity_tracker.update_state(perplexity)

        return {m.name: m.result() for m in self.metrics}


    def test_step(self, inputs):
        """
        Runs a forward pass without gradient updates and computes evaluation loss.
        Updates accuracy and perplexity using the same masked next-token objective.
        Returns the current values of all tracked evaluation metrics.
        """
        input_ids = inputs['input_ids']
        loss_mask = inputs['loss_mask']
 
        y_pred = self(inputs, training=False)
        loss = self.compute_loss(input_ids, y_pred, loss_mask)
        
        # Compute metrics
        accuracy = self.compute_accuracy(input_ids, y_pred, loss_mask)
        perplexity = tf.math.exp(loss)

        # Update loss and metrics trackers
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        self.perplexity_tracker.update_state(perplexity)
        
        # Return metrics
        return {m.name: m.result() for m in self.metrics}
    
    
    # Register trackers
    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker, self.perplexity_tracker]
