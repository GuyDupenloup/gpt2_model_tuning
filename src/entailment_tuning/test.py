
# Copyright (c) 2025 Guy Dupenloup
# Licensed under the MIT License. See LICENSE file for details.

from utils.model_utils import create_gpt2_classifier, print_model_variables
from utils.classify import classify_prompts

class_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

model_size = '124M'

print(f'>> Creating GPT-2 model `{model_size}`')
model = create_gpt2_classifier(model_size, num_classes=len(class_names))
print_model_variables(model)

# Example prompt
prompt = 'The secret to living a happy life is'
print(f'\n>> Prompt:\n{prompt}')

predictions = classify_prompts(
    model,
    [prompt]   # The function takes a list of prompts.
)

predicted_class = predictions[0]
print(f'\n>> Class:  {class_names[predicted_class]}')
