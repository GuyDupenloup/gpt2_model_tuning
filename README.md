# GPT-2 Model Tuning

## 1. Introduction

In this project, I fine-tune a GPT-2 model for three goals:
1. instruction tuning
2. classification tuning
3. entailment tuning

LoRA is built-in the model and was used for entailment tuning.

The project uses the Tensorflow GPT-2 model that I developed in a previous project from scratch, using only research papers. See the 'gpt2_model_from_research_papers' repo.

## 2. Source

The source code for this project in the ./src directory. There are 4 different subdirectories:

```
    src
     |     
     ├── common
     |     |
     |     ├── gpt2_model.py                     # GPT-2 base model
     |     └── model_utils.py                    # Model utilities and pretrained weights transfer
     |
     ├── instruction_tuning
     |     |
     |     ├── prep_dataset.py                   # Format dataset and write TFRecords
     |     ├── gpt2_textgen_model.py             # GPT-2 model with text generation head
     |     ├── train.py                          # Train the model
     |     └── gen_text.py                       # Text generation from a prompt
     |
     ├── classification_tuning
     |     |
     |     ├── prep_dataset.py                   # Format dataset and write TFRecords
     |     ├── gpt2_classification_model.py      # GPT-2 model with classification head
     |     └── train.py                          # Train the model
     |
     └── entailment_tuning_with_lora
           |
           ├── prep_dataset.py                   # Format dataset and write TFRecords
           ├── gpt2_entailment_model.py          # GPT-2 model with entailment classification head
           └── train.py                          # Train the model

```

## 3. GPT-2 Model


