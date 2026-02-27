
# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

from utils.model_utils import create_gpt2_language_model, print_model_variables
from utils.gen_text import generate_text

model_size = '124M'

print(f'>> Creating GPT-2 model `{model_size}`')
model = create_gpt2_language_model(model_size)
print_model_variables(model)

# Example prompt
prompt = 'The secret to living a happy life is'
print(f'\n>> Prompt:\n{prompt}')

output_text = generate_text(
    model,
    [prompt],   # The function takes a list of prompts.
    output_len=50,
    sampling_method='top_p',
    temperature=0.8,
    top_k=20,
    top_p=0.9
)
print(f'\n>> Output text:\n{output_text[0]}')
