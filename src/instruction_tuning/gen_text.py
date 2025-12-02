# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.
 
import numpy as np
import tensorflow as tf
import tiktoken
from models.model_utils import create_gpt2_language_model


def sample_next_token(logits, sampling_method='greedy', temperature=1.0, top_k=0, top_p=1.0):
    """
    Sample the next token from a language model's logits using different sampling methods.
    
    Parameters:
        logits: 1D numpy array of model logits for the current step
        sampling_method: 'greedy', 'temperature', 'top_k', or 'top_p'
        temperature: float > 0, used for temperature scaling
        top_k: integer >= 1, number of top-k tokens to consider (only for top_k sampling)
        top_p: float in (0,1], cumulative probability threshold for nucleus sampling (top-p)
    
    Returns:
        next_token_id: int, index of the sampled token
    """

    # Define numerically stable softmax
    def softmax(x):
        x = x.astype(np.float64)  # ensure stability for large logits
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    if sampling_method == 'greedy':
        # Take the token with the highest logit
        next_token_id = np.argmax(logits)

    elif sampling_method == 'temperature':
        # Convert logits to probabilities and sample
        probs = softmax(logits / temperature)
        next_token_id = np.random.choice(len(probs), p=probs)

    elif sampling_method == 'top_k':
        # Apply temperature scaling to logits
        logits = logits / temperature

        # Get indices and values of top-k logits
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        top_k_logits = logits[top_k_indices]

        # Convert top-k logits to probabilities and sample
        top_k_probs = softmax(top_k_logits)
        next_token_id = np.random.choice(top_k_indices, p=top_k_probs)
    
    elif sampling_method == 'top_p':
        # Sort logit probabilities in descending order
        probs = softmax(logits)
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]

        # Calculate cumulative probabilities
        cumsum_probs = np.cumsum(sorted_probs)

        # Find cutoff index where cumulative probability exceeds top_p
        cutoff_idx = np.searchsorted(cumsum_probs, top_p)
        cutoff_idx = max(1, cutoff_idx)  # ensure at least one token is included

        # Keep only top-p tokens
        top_p_indices = sorted_indices[:cutoff_idx]
        top_p_probs = sorted_probs[:cutoff_idx]

        # Renormalize probabilities for the filtered tokens
        top_p_probs = top_p_probs / np.sum(top_p_probs)
        next_token_id = np.random.choice(top_p_indices, p=top_p_probs)

    return int(next_token_id)


def check_next_token_sampling_params(sampling_method, temperature, top_k, top_p):
    """
    Checks that next-token sampling parameters are correctly set
    """

    if sampling_method not in ('greedy', 'temperature', 'top_k', 'top_p'):
        raise ValueError("Supported sampling methods are 'greedy', 'temperature', 'top_k', and 'top_p'")
    
    if  sampling_method in ('temperature', 'top_k', 'top_p'):
        if temperature <= 0:
            raise ValueError('temperature argument must be > 0')
        
    if sampling_method == 'top_k':
        if top_k < 1:
            raise ValueError('top-k argument must be >= 1')
        
    if sampling_method == 'top_p':
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError('top-p argument must be > 0.0 and <= 1.0')
            

def generate_text(
    model: 'tf.keras.Model', 
    prompt: str, 
    output_len: int, 
    sampling_method: str = 'greedy', 
    temperature: float = 1.0, 
    top_k: int = 1, 
    top_p: float = 1.0
) -> str: 
   
    """
    Generates an output text given an input prompt and a max number of output tokens

    Arguments:
        model: 
            GPT-2 Keras model
        prompt: input text to start from
        output_len: maximum number of output tokens (including the prompt)
        sampling_method: 'greedy', 'temperature', 'top_k', or 'top_p'
        temperature: float > 0, used for temperature scaling
        top_k: integer >= 1, number of top-k tokens to consider for top-k sampling
        top_p: float in (0.0, 1.0], cumulative probability threshold to use for nucleus sampling (top-p)

    Returns:
        Output text

    Sampling methods:
    ----------------
        'greedy':
            The token with the largest logit is selected (deterministic).
        'temperature':
            Logits scaled by temperature are converted to probabilities and a token
            is sampled from the distribution.
        'top_k':
            Only the top-k tokens (based on scaled logits) are considered. They are
            renormalized and a token is sampled from the distribution.
        'top_p':
            Tokens are sorted by probability (from scaled logits). The smallest set of
            tokens whose cumulative probability sum is >= top_p are kept, renormalized,
            and a token is sampled from this nucleus.
    """

    check_next_token_sampling_params(sampling_method, temperature, top_k, top_p)

    # GPT-2 context length and padding token
    seq_len = 1024
    pad_token = 50256

    # Encode the prompt
    tokenizer = tiktoken.get_encoding('gpt2')
    tokens_out = tokenizer.encode(prompt)

    for _ in range(output_len):
        current_tokens = tokens_out[-seq_len:]
        num_pad = seq_len - len(current_tokens)

        # Pad input tokens to the left
        if num_pad > 0:
            padded_input = [pad_token] * num_pad + current_tokens
            attention_mask = [0] * num_pad + [1] * len(current_tokens)
        else:
            padded_input = current_tokens
            attention_mask = [1] * seq_len

        # Run the model
        inputs = {
            'input_ids': tf.constant([padded_input], dtype=tf.int32),
            'attention_mask': tf.constant([attention_mask], dtype=tf.int32)
        }
        hidden_states = model(inputs)

        # Get the last hidden state, convert to numpy
        # and get rid of the batch dimension
        logits = hidden_states[:, -1, :]
        logits = np.squeeze(logits.numpy())

        # Sample the next token and append it to outout
        next_token = sample_next_token(
            logits,
            sampling_method=sampling_method,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        tokens_out.append(int(next_token))

    # Decode the output text
    output_text = tokenizer.decode(tokens_out)

    return output_text


# To be replaced by the fine-tuned model
model = create_gpt2_language_model('gpt2')

# Example prompt
prompt = 'The future of artificial intelligence is'

print(f'\n>> Input text:\n{prompt}')
output_text = generate_text(
    model,
    prompt,
    output_len=50,
    sampling_method='top_p',
    temperature=1.0,
    top_p=0.9
)
print(f'\n>> Output text:\n{output_text}')
