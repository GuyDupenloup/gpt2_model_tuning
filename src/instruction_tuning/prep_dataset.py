# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

import os
import argparse
import json
from glob import glob
import numpy as np
import tensorflow as tf
from datasets import load_dataset
import tiktoken


def tokenize_dataset(data_ds, tokenizer, seq_len=1024, pad_token=50256):

    input_ids = []
    attention_mask = []
    loss_mask = []

    truncated = 0

    for example in data_ds:
        instruction = example['instruction']
        context = example['context'] if 'context' in example else ''
        response = example['response']

        # Create the prompt (there may be no context)
        if context:
            prompt_text = f'### Instruction: {instruction}\n\n### Context: {context}\n\n### Response: {response}'
            prefix_text = f'### Instruction: {instruction}\n\n### Context: {context}'
        else:
            prompt_text = f'### Instruction: {instruction}\n\n### Response: {response}'
            prefix_text = f'### Instruction: {instruction}'

        # Tokenize the prompt
        prompt_ids = tokenizer.encode(prompt_text)

        # Drop the example if the prompt requires 
		# more tokens than the sequence length
        if len(prompt_ids) > seq_len:
            prompt_ids = prompt_ids[:seq_len]
            truncated += 1

        # Pad the prompt to sequence length
        for _ in range(len(prompt_ids), seq_len):
            prompt_ids.append(pad_token)
        input_ids.append(prompt_ids)

        # Tokenize the prefix to get its length
        prefix_ids = tokenizer.encode(prefix_text)
        prefix_len = len(prefix_ids)

        # Create the attention mask
        attention_mask.append(
            [0 if prompt_ids[i] == pad_token else 1 for i in range(seq_len)]
        )

        # Create the loss mask
        loss_mask.append(
            [0 if i < prefix_len else 1 for i in range(seq_len)]
        )

    if truncated > 0:
        print(f'Truncated {truncated} examples using more than {seq_len} tokens')
        
    # Convert from lists to numpy arrays
    input_ids = np.array(input_ids, dtype=np.int32)
    attention_mask = np.array(attention_mask, dtype=np.int32)
    loss_mask = np.array(loss_mask, dtype=np.int32)

    # Wrap outputs in dictionary
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask
    }


def write_tfrecord(data, filepath):

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    loss_mask = data['loss_mask']

    def _int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    num_examples = input_ids.shape[0]
    with tf.io.TFRecordWriter(filepath) as writer:
        for i in range(num_examples):
            feat = {
                'input_ids': _int_feature(input_ids[i].astype(np.int64)),
                'attention_mask': _int_feature(attention_mask[i].astype(np.int64)),
                'loss_mask': _int_feature(loss_mask[i].astype(np.int64))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(example.SerializeToString())


def parse_and_write_dataset(output_dir):

    # Load Databricks' Dolly 15k dataset
    dataset_name = 'databricks/databricks-dolly-15k'
    print(f'Loading dataset `{dataset_name}`')
    dataset = load_dataset(dataset_name, split='train')

    # Split into train (80%) and temp (20%)
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = train_test['train']
    temp_ds = train_test['test']

    # Split temp into val (50%) and test (50%)
    # This gives 80% train, 10% val, 10% test
    val_test = temp_ds.train_test_split(test_size=0.5, seed=42)
    val_ds = val_test['train']
    test_ds = val_test['test']

    # Load GPT-2 tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')

    print('\nTokenizing training set')
    train_data = tokenize_dataset(train_ds, tokenizer)
    train_size = train_data['input_ids'].shape[0]
    print('Size:', train_size)

    print('\nTokenizing validation set')
    val_data = tokenize_dataset(val_ds, tokenizer)
    val_size = val_data['input_ids'].shape[0]
    print('Size:', val_size)

    print('\nTokenizing test set')
    test_data = tokenize_dataset(test_ds, tokenizer)
    test_size = test_data['input_ids'].shape[0]
    print('Size:', test_size)

    # Create output directory if it does not exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Save dataset metadata to JSON file
    metadata = {
        'dataset_name': 'Databricks Dolly 15k',
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size
    }
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print('\nWriting dataset to TFRecords')
    write_tfrecord(train_data, filepath=os.path.join(output_dir, 'train.tfrecord'))
    write_tfrecord(val_data, filepath=os.path.join(output_dir, 'val.tfrecord'))
    write_tfrecord(test_data, filepath=os.path.join(output_dir, 'test.tfrecord'))

    print('Done')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        help='Output directory where to store TFRecords',
        type=str,
        default='./dataset'
    )
    
    args = parser.parse_args()
    parse_and_write_dataset(args.output_dir)
