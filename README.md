# GPT-2 Model Tuning

## 1. Introduction

In a previous project, I created a GPT-2 model from scratch using the original Transformer paper and the GPT/GPT-2 papers from OpenAI.

See Github repo: [GPT-2 Model From Research Papers:](https://github.com/GuyDupenloup/gpt2_model_from_research_papers)

In this project, I fine-tune that model for three different applications:

- **Instruction tuning**: teach the model to follow instructions.

- **Classification tuning**: categorize pieces of text into predefined classes.

- **Entailment tuning**: determine whether the meaning of one sentence implies the truth of another.

The project also includes an implementation of LoRA, a highly effective method for parameter-efficient fine-tuning.


## 2. Source code and Python packages

The source code for this project is in the *./src* directory and is organized as shown below.

```
    src
     |     
     ├── common
     |     |
     |     ├── gpt2_model.py                     # GPT-2 base model (no output linear layer) with built-in LoRA layers
     |     ├── gpt2_language_model.py            # GPT-2 language model (base model with LM head)
     |     ├── gpt2_classifier.py                # GPT-2 classifier (base model with classification head)
     |     └── model_utils.py                    # Model utilities (get GPT-2 configurations, create models, print model variables)
     |
     ├── instruction_tuning
     |     |
     |     ├── prep_dataset.py                   # Dataset parsing and TFRecords export
     |     ├── train.py                          # Model tuning
     |     └── gen_text.py                       # Text generation from a starting prompt
     |
     ├── classification_tuning
     |     |
     |     ├── prep_dataset.py                   # Dataset parsing and TFRecords export
     |     └── train.py                          # Model tuning
     |
     └── entailment_tuning_with_lora
           |
           ├── prep_dataset.py                   # Dataset parsing and TFRecords export
           └── train.py                          # Model tuning

```

See file *requirements.txt* for the list of Python packages I used.

## 3. GPT-2 base Model

I made two important additions to my GPT-2 original model:

- An attention mask so the model does not attend to padding tokens.

- Built-in Low-Rank Adaptation (LoRA) layers inside the multi-head attention blocks, which can be enabled or left inactive depending on the configuration.

Other changes include the removal of the language modelling output layer, leaving only the GPT-2 base model.

The LoRA implementation follows the method described in the original paper. LoRA layers are inserted into the existing weight matrices within the multi-head attention blocks. These layers introduce two small, low-rank matrices which are the only parameters updated during fine-tuning. The original, large pre-trained weights remain frozen, significantly reducing the number of trainable parameters and making the fine-tuning process far more efficient.

Hugging Face's *TFGPT2LMHeadModel* from the *transformers* package is used to obtain OpenAI's pretrained weights. As the two models strictly followed the architecture described in the research papers, transferring the weights is straightforward. See the README of the model creation project for details.

## 4. Dataset preparation

Each model tuning directory contains a script called *prep_dataset.py*.

This script loads the dataset, assembles the prompt, tokenizes it, and truncates or pads the token list to a fixed length of 1024 (the GPT-2 context size). It then exports the training, validation, and test sets to TFRecords for optimized data loading and pipelining in TensorFlow.

Token ID 50256 is used for padding, and the attention mask ensures that padding tokens are ignored by the model.

To change the default output directory name or TFRecord filenames, use the *--help* option of the script.


## 5. Running scripts

### 5.1 Setting Python's search path

To run scripts, you need to add the *src* directory (absolute path) to the PYTHONPATH environment variable that sets the search path for Python.

To set it in Linux:

```
export PYTHONPATH="/path/to/src/dir:$PYTHONPATH"
```

To set it in Windows:
```

# powershell (permanent)
$env:PYTHONPATH = "$env:PYTHONPATH;C:\path\to\src\dir"

# cmd (temporary)
set PYTHONPATH=%PYTHONPATH%;C:\path\to\src\dir

```

### 5.2 Running dataset preparation and training scripts

To run a dataset preparation script or a training script, first change the current directory to the directory where it is located. Then, execute the script.

The dataset preparation script must be run before running the training script to create TFRecords for the training, validation and test sets.

For example, to tune the classification GPT-2 model:

```
# Go to classification tuning directory
cd src/classification_tuning

# Prepare the dataset
python prep_dataset.py

# Tune the classification model
python train.py
```

## 6. Instruction tuning

Instruction tuning uses the **Databricks Dolly-15k** dataset. Each example contains two or three fields: instruction, context for some examples, and response.

The dataset preparation script concatenates these fields into a single prompt, inserting the headers "### Instruction:", "### Context:", and "### Response:" as separators.

A sample prompt is shown below:

```
### Instruction: In Frank Herbert's Dune novel, why is the spice valuable?

### Context: Sandworms are colossal, worm-like creatures that live on the desert planet Arrakis. The sandworms' larvae produce a drug called melange (known colloquially as "the spice"), the most essential and valuable commodity in the universe because it makes safe and accurate interstellar travel possible. Melange deposits are found in the sand seas of Arrakis, where the sandworms live and hunt, and harvesting the spice from the sand is a dangerous activity because sandworms are aggressive and territorial. Harvesting vehicles must be airlifted in and out of the sand sea in order to evade sandworm attacks. The struggle over the production and supply of melange is a central theme of the Dune saga.

### Response: The spice is valuable because it is a scarce resource that is crucial to interstellar travel. The spice is scarce because it can be found only on planet Arrakis, and its extraction is difficult due to the presence of sandworms.
```
A language modelling head is added to the base model for this application (as described in OpenAI's GPT/GPT-2 papers).

During training, the loss is computed only on the response portion of the prompt. The model must use the instruction and context portions as inputs, but must not be penalized for predicting them.

A script is provided to generate text from a starting prompt (*gen_text.py* in *instruction_modeling* directory). It supports four decoding strategies: greedy, temperature scaling, top-k sampling, and top-p (nucleus) sampling. You can modify the script to try your own prompt or use a different model.

## 7. Classification tuning

Classification tuning uses the **Hugging Face emotion** dataset.

Examples from this dataset are given in the table below.

| Text                                                                                   |  Label    |
|----------------------------------------------------------------------------------------|-----------|
| i feel so pathetic and useless being unable to do anything                             |  sadness  |
| i feel that i am useful to my people and that gives me a great feeling of achievement  |  joy      |
| i just want to feel loved by you                                                       |  love     |
| i feel more aggravated and annoyed by their visits                                     |  anger    |
| i was feeling a little fearful of trying to eat this damn thing                        |  fear     |
| i feel like i should not be surprised at this development                              | surprise  |

A 6-class classification head (dense layer) is added to the GPT-2 base model for this application.


## 8. Entailment tuning using LoRA

Entailment tuning uses the **Hugging Face glue** dataset. Each example contains two sentences and a label indicating whether the first sentence entails the second.

Example without entailment:

```
Sentence 1: "About 3 million years ago, when Lucy was alive, she was rather short, about 4 feet tall, and probably weighed about 50 pounds."
Sentence 2: "Humans existed 10,000 years ago."
```

Example with entailment:

```
 Sentence 1: "The abode of the Greek gods was on the summit of Mount Olympus, in Thessaly."
 Sentence 2: "Mount Olympus is in Thessaly."
```

A 2-class classification head (dense layer) is added to the GPT-2 base model for this task.

Token ID 50256 is used to separate the two sentences, serving as a delimiter since the standard GPT-2 vocabulary lacks a dedicated separator token. Unlike padding tokens, these are not masked.

LoRA is used for fine-tuning. Since LoRA layers are built into the GPT-2 base model, only two steps are required:

- Provide a LoRA configuration dictionary to activate the layers and set their rank and alpha parameters.

- Call the model’s freeze_all_but_lora() method to make all other layers non-trainable.

## 9. Training results

Tuning QoR essentially depends on the GPU resources you have at your disposal...

If you just want to test the training scripts, you can use the smallest GPT-2 model and take a fraction of the dataset. You should be able to see how the loss decreases and accuracy increases over a few epochs. This is doable using the free version of Google Colab.
