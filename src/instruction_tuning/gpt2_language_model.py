# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import tensorflow as tf
from common.gpt2_model import GPT2Model


class GPT2TextGenModel(tf.keras.models.Model):

    def __init__(self, model_config, lora_config=None, dropout_rate=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.config = model_config
        self.gpt2_model = GPT2Model(model_config, lora_config=lora_config, dropout_rate=dropout_rate, name=name)

        if lora_config is not None:
            self.gpt2_model.freeze_all_but_lora()

        # Loss and metrics trackers
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.accuracy_tracker = tf.keras.metrics.Mean(name='accuracy')
        self.perplexity_tracker =  tf.keras.metrics.Mean(name='perplexity')


    def call(self, data):

        # data = {'input_ids': input_ids, 'attention_mask': attention_mask}
        gpt2_output = self.gpt2_model(data)

        # Text prediction linear layer (no trainable weights)
        embedding_weights = self.gpt2_model.token_embed_layer.embeddings
        logits = tf.matmul(gpt2_output, embedding_weights, transpose_b=True)

        return logits


    def compute_loss(self, input_ids, y_pred, mask):

        # Shift inputs to get labels
        y_true = input_ids[:, 1:]

        # Truncate predictions and shift mask to align with labels
        y_pred = y_pred[:, :-1, :]
        mask = mask[:, 1:]
        
        # Calculate cross-entropy loss per token (element wise)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        
        # Apply mask
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = loss * mask
        
        # Return mean loss over non-masked tokens
        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


    def compute_accuracy(self, input_ids, y_pred, mask):

        # Shift inputs to get labels
        y_true = input_ids[:, 1:]

        # Truncate predictions and shift mask to align with labels
        y_pred = y_pred[:, :-1, :]
        mask = mask[:, 1:]
        
        # Get predicted token IDs (argmax over vocabulary dimension)
        predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)  # Shape: (batch_size, seq_len - 1)
        
        # Compare predictions with true labels (1.0 if correct, 0.0 if wrong)
        correct = tf.cast(tf.equal(predictions, y_true), dtype=tf.float32)
        
        # Apply mask
        mask = tf.cast(mask, dtype=tf.float32)
        correct_masked = correct * mask
        
        # Calculate accuracy: correct predictions / total non-masked tokens
        accuracy = tf.reduce_sum(correct_masked) / tf.maximum(tf.reduce_sum(mask), 1.0)
        
        return accuracy


    def train_step(self, data):
        # There are no labels as they are obtained by shifting the inputs.
        input_ids = data['input_ids']
        loss_mask = data['loss_mask']
 
        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)  # Pass full x dict to call()
            loss = self.compute_loss(input_ids, y_pred, loss_mask)
    
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Compute metrics
        accuracy = self.compute_accuracy(input_ids, y_pred, loss_mask)
        perplexity = tf.math.exp(loss)

        # Update loss and metrics trackers
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        self.perplexity_tracker.update_state(perplexity)
        
        # Return metrics
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        input_ids = data['input_ids']
        loss_mask = data['loss_mask']
 
        y_pred = self(data, training=False)
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
