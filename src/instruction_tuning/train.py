# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import json
import argparse
from timeit import default_timer as timer
from datetime import timedelta
import tensorflow as tf

from utils.model_utils import create_gpt2_language_model, print_model_variables


def load_dataset(dataset_dir):

    # Read the metadata JSON file that contains the dataset sizes
    metadata_path = os.path.join(dataset_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f">> Loading dataset `{metadata['dataset_name']}`:")
    print(f"  train size: {metadata['train_size']}")
    print(f"  val size: {metadata['val_size']}")
    print(f"  test size: {metadata['test_size']}")

    # Read dataset TFRecords
    train_record = tf.data.TFRecordDataset(os.path.join(dataset_dir, 'train.tfrecord'))
    val_record = tf.data.TFRecordDataset(os.path.join(dataset_dir, 'val.tfrecord'))
    test_record = tf.data.TFRecordDataset(os.path.join(dataset_dir, 'test.tfrecord'))

    return train_record, val_record, test_record


def create_data_loader(ds, batch_size, seq_len=1024, pad_token=50256, shuffle=False, buffer_size=1000):
    """
    Creates a tf.data.Dataset pipeline.
    """
    def parse_and_wrap(example_proto):

        feature_spec = {
            'input_ids': tf.io.FixedLenFeature([seq_len], tf.int64),
            'attention_mask': tf.io.FixedLenFeature([seq_len], tf.int64),
            'loss_mask': tf.io.FixedLenFeature([seq_len], tf.int64),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_spec)

        return {
            'input_ids': tf.cast(parsed['input_ids'], tf.int32),
            'attention_mask': tf.cast(parsed['attention_mask'], tf.int32),
            'loss_mask': tf.cast(parsed['loss_mask'], tf.int32)
        }

    ds = ds.map(parse_and_wrap, num_parallel_calls=tf.data.AUTOTUNE)
    # ds = ds.cache() 
    if shuffle:
        ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


def train_model(model_size, dataset_dir, output_dir):

    # Set output file paths
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    metrics_csv_path = os.path.join(output_dir, 'metrics.csv')

    # Read dataset TFRecords
    if not os.path.isdir(dataset_dir):
        raise ValueError(f'Unable to find dataset directory {dataset_dir}')
    train_record, val_record, test_record = load_dataset(dataset_dir)

    # Create data loaders
    print('>> Creating data loaders')
    batch_size = 4
    train_ds = create_data_loader(train_record, batch_size=batch_size, shuffle=True)
    val_ds = create_data_loader(val_record, batch_size=batch_size)
    test_ds = create_data_loader(test_record, batch_size=batch_size)

    # Get the model with pretrained weights
    print(f'>> Creating pretrained language model `{model_size}`')
    model = create_gpt2_language_model(model_size, dropout_rate=0.1)

    model.save_weights('pretrained_weights.h5')

    print_model_variables(model, trainable=True, non_trainable=True, params_only=True)

    # Compile the model
    # Don't pass loss or metrics, let the model handle it.
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0)
    model.compile(optimizer=optimizer)

    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoints_dir, 'checkpoint_{epoch:02d}.weights.h5'),
            save_weights_only=True,
            save_best_only=False,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.CSVLogger(
            filename=metrics_csv_path
        )
    ]

    # Train the model
    print('>> Starting training')
    epochs = 3
    start_time = timer()
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    end_time = timer()

    train_run_time = int(end_time - start_time)
    print('>> Training runtime: ' + str(timedelta(seconds=train_run_time))) 

    # Load the last weights obtained in the training
    last_weights_path = os.path.join(checkpoints_dir, f'checkpoint_{epochs}.weights.h5')
    model.load_weights(last_weights_path)

    # Evaluate the model on test set
    print('>> Evaluating fine-tuned model on test set')
    loss, accuracy, perplexity = model.evaluate(test_ds, verbose=1)
    print(f' loss: {loss:.4f}')
    print(f' accuracy: {accuracy:.4f}')
    print(f' perplexity: {perplexity:.4f}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_size',
        help='GPT-2 model size',
        type=str,
        choices=['124M', '355M', '774M', '1542M'],
        default='124M'
    )
    parser.add_argument(
        '--dataset_dir',
        help='Directory where the dataset TFRecords are',
        type=str,
        default='./dataset'
    )
    parser.add_argument(
        '--output_dir',
        help='Directory where to save training output files (checkpoint, trained model, etc.)',
        type=str,
        default='./train_output'
    )
    
    args = parser.parse_args()
    train_model(args.model_size, args.dataset_dir,args.output_dir)
