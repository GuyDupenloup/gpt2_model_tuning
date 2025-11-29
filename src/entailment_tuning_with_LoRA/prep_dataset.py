# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.
 
import os
import argparse
import json
import numpy as np
import tensorflow as tf
from datasets import load_dataset
import tiktoken


def tokenize_dataset(data_ds, tokenizer, seq_len=1024, pad_token=50256):

    classes = {}
    input_ids = []
    labels = []
    dropped = 0
	
    for example in data_ds:
        sentence1 = example['sentence1']
        sentence2 = example['sentence2']

        # Tokenize the sentences
        sentence1_ids = tokenizer.encode(sentence1)
        sentence2_ids = tokenizer.encode(sentence2)

        # The padding token is used as a separator.
        text_ids = sentence1_ids + [pad_token] + sentence2_ids

        # Drop the example if the text
		# requires more tokens than max_length
        if len(text_ids) > seq_len:
            dropped += 1
            continue

        # Pad the text to sequence length
        for i in range(len(text_ids), seq_len):
            text_ids.append(pad_token)

        classes[example['label']] = True

        # Append to lists of all examples
        input_ids.append(text_ids)
        labels.append(example['label'])

    if dropped > 0:
        print(f'Dropped {dropped} examples using more than {seq_len} tokens')

    # Convert from lists to numpy arrays
    input_ids = np.array(input_ids, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)

    # Wrap outputs in dictionary
    data = {'input_ids': input_ids, 'labels': labels}
    
    return data, len(classes)


def write_tfrecord(data, filepath):

    input_ids = data['input_ids']
    labels = data['labels']

    def _int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    num_examples = input_ids.shape[0]
    with tf.io.TFRecordWriter(filepath) as writer:
        for i in range(num_examples):
            feat = {
                'input_ids': _int_feature(input_ids[i].astype(np.int64)),
                'labels': _int_feature([int(labels[i])])  # scalar -> list
            }
            example = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(example.SerializeToString())


def parse_and_write_dataset(output_dir):

    # Load Hugging Face RTE dataset (Recognizing Textual Entailment)
    dataset = load_dataset('glue', 'rte')

    train_ds = dataset['train']
    val_ds = dataset['validation']
    test_ds = dataset['test']

    print('Dataset size:')
    print(f' training: {len(train_ds)}')
    print(f' validation: {len(val_ds)}')
    print(f' test: {len(test_ds)}')

    # Load GPT-2 tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    # Tokenize training set
    print('\nTokenizing training set')
    train_data, num_classes = tokenize_dataset(train_ds, tokenizer)
    train_size = train_data['input_ids'].shape[0]
    print('Classes:', num_classes)
    print('Size:', train_size)

    # Tokenize validation set
    print('\nTokenizing validation set')
    val_data, _ = tokenize_dataset(val_ds, tokenizer)
    val_size = val_data['input_ids'].shape[0]
    print('Size:', val_size)

    # Tokenize test set
    print('\nTokenizing test set')
    test_data, _ = tokenize_dataset(test_ds, tokenizer)
    test_size = test_data['input_ids'].shape[0]
    print('Size:', test_size)

    # Create output directory if it does not exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Save dataset metadata to JSON file
    metadata = {
        'dataset_name': 'Hugging Face Glue (RTE)',
        'num_classes': num_classes,
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
