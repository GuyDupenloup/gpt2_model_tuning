# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.
 
import tensorflow as tf
import tiktoken


def classify_prompts(
    model: 'tf.keras.Model',
    prompts: list
) -> list:
    
    """
    """

    seq_len = 1024
    pad_token = 50256
    tokenizer = tiktoken.get_encoding('gpt2')

    # Encode and truncate/pad all prompts, set attention mask
    input_ids = []
    attention_mask = []

    for text in prompts:
        tokens = tokenizer.encode(text)   # Encode
        tokens = tokens[:seq_len]   # Truncate
        tokens += [pad_token for _ in range(len(tokens), seq_len)]   # Pad

        attn = [0 if tokens[i] == pad_token else 1 for i in range(seq_len)]

        input_ids.append(tokens)
        attention_mask.append(attn)

    # Run prompts through the model
    inputs = {
        'input_ids': tf.convert_to_tensor(input_ids, dtype=tf.int32),
        'attention_mask': tf.convert_to_tensor(attention_mask, dtype=tf.int32)
    }
    
    logits = model(inputs)

    predicted_classes = tf.argmax(logits, axis=-1)

    return predicted_classes.numpy().tolist()
