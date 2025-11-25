# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import shutil
import json
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import tensorflow as tf

from gpt2_classification_model import GPT2ClassificationModel
from common.model_utils import get_gpt2_model_config, get_pretrained_weights

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


def load_dataset(dataset_dir):

    # Read the JSON file that contains the dataset sizes
    metadata_path = os.path.join(dataset_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Loading dataset `{metadata['dataset_name']}`:")
    print(f"  classes: {metadata['num_classes']}")
    print(f"  train size: {metadata['train_size']}")
    print(f"  val size: {metadata['val_size']}")
    print(f"  test size: {metadata['test_size']}")

    # Read dataset TFRecords
    train_record = tf.data.TFRecordDataset(os.path.join(dataset_dir, 'train.tfrecord'))
    val_record = tf.data.TFRecordDataset(os.path.join(dataset_dir, 'val.tfrecord'))
    test_record = tf.data.TFRecordDataset(os.path.join(dataset_dir, 'test.tfrecord'))

    return metadata['num_classes'], train_record, val_record, test_record


def create_data_loader(ds, batch_size, seq_len=1024, pad_token=50256, shuffle=False, buffer_size=1000):
    """
    Creates a tf.data.Dataset pipeline.
    """
    def parse_and_wrap(example_proto):
            
            # Parse TFRecord
            feature_spec = {
                'input_ids': tf.io.FixedLenFeature([seq_len], tf.int64),
                'labels': tf.io.FixedLenFeature([], tf.int64),
            }
            parsed = tf.io.parse_single_example(example_proto, feature_spec)

            # Convert from tf.int64 to tf.int32
            input_ids = tf.cast(parsed['input_ids'], tf.int32)
            labels = tf.cast(parsed['labels'], tf.int32)
            
            # Create the attention mask
            attention_mask = [0 if input_ids[i] == pad_token else 1 for i in range(seq_len)]

            # Wrap model inputs in dictionary
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }

            return model_inputs, labels


    ds = ds.map(parse_and_wrap, num_parallel_calls=tf.data.AUTOTUNE)
 
    if shuffle:
        ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


def create_classification_model(model_size, num_classes):

    model_config = get_gpt2_model_config(model_size)
    model = GPT2ClassificationModel(model_config, num_classes, dropout_rate=0.1, name='gpt2_classify')

    # Build the model using dummy inputs
    seq_len = model_config['seq_len']
    vocab_size = model_config['vocab_size']
    dummy_input = {
        'input_ids': tf.random.uniform((1, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32),
        'attention_mask': tf.random.uniform((1, seq_len), minval=0, maxval=2, dtype=tf.int32)
    }
    _ = model(dummy_input)

	# Load Hugging Face model pretrained variables 
    # for a GPT-2 model of the same size
    pretrained_weights = get_pretrained_weights(model_size)

	# Check that the two models have the same 
	# number of trainable variables
    num_vars = len(pretrained_weights) 
    assert len(model.trainable_variables) == num_vars

    for i in range(num_vars):
        var = model.trainable_variables[i]
        weights = var.numpy()
        weights_pt = pretrained_weights[i]

        assert weights.shape == weights_pt.shape
        var.assign(weights_pt)

    return model


# Read dataset TFRecords
num_classes, train_record, val_record, test_record = load_dataset('/content/dataset')

# Create data loaders
train_ds = create_data_loader(train_record, batch_size=2, shuffle=True)
val_ds = create_data_loader(val_record, batch_size=2)
test_ds = create_data_loader(test_record, batch_size=2)

# #Take a fraction of the dataset
train_ds = train_ds.take(500)
val_ds = val_ds.take(100)
test_ds = train_ds.take(100)

# Get the model with pretrained weights
model_name = '117M'
print(f'Creating classification model `{model_name}`')
model = create_classification_model(model_name, num_classes)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# Set output file paths
output_dir = '/content/train_output'   # For Google Colab
checkpoint_path = os.path.join(output_dir, 'checkpoint.weights.h5')
tensorboard_logs = os.path.join(output_dir, 'tensorboard_logs')
metrics_csv_path = os.path.join(output_dir, 'metrics.csv')
tuned_weights_path = os.path.join(output_dir, 'tuned_model.weights.h5')
config_path = os.path.join(output_dir, 'model_config.json')

# Prepare training output dir
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

# Set up callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='auto'
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_logs,
        update_freq='epoch'
    ),
    tf.keras.callbacks.CSVLogger(
        filename=metrics_csv_path
    )
]

# Train model
print('Starting training...')
start_time = timer()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    callbacks=callbacks,
    verbose=1
)
end_time = timer()
train_run_time = int(end_time - start_time)
print("Training runtime: " + str(timedelta(seconds=train_run_time))) 

# Load best weights obtained in the training
print('Loading best weights')
model.load_weights(checkpoint_path)

# Evaluate the model on test set
print('Evaluating fine-tuned model on test set')
loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f' loss: {loss:.4f}')
print(f' accuracy: {accuracy:.4f}')

# Save model config and tuned weights
print(f'Saving fine-tuned model in {output_dir}')
with open(config_path, 'w') as f:
    json.dump(model.config, f, indent=2)
model.save_weights(tuned_weights_path)
