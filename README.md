# Transformer in C

![C](https://img.shields.io/badge/language-C-blue)
![CUDA](https://img.shields.io/badge/GPU-CUDA-green)
![License](https://img.shields.io/badge/license-MIT-orange)

A lightweight, from-scratch Transformer implementation in C, designed for educational purposes and performance experimentation, with optional CUDA acceleration.

---

## Features
- Pure C implementation (no frameworks)
- Byte Pair Encoding (BPE) Tokenizer
- Training on custom datasets (Project Gutenberg)
- CPU and CUDA (GPU) versions
- Configurable model sizes

---

## Why this project?

This project was built to:
- Understand transformers at a low level (without frameworks)
- Explore performance trade-offs between CPU and GPU
- Experiment with small-scale language models

---

## Requirements

**CPU version:**
```bash
sudo apt install gcc libopenblas-dev
```

**GPU version:**
- NVIDIA GPU with CUDA Toolkit installed, or
- Google Colab (free, no installation needed)

---

## Project structure

```
c-transformer/
├── src/           # C source code
├── include/       # Header files
├── build/         # Compiled binaries
├── models/        # Trained weights and tokenizer files
│   ├── DIM128/   
│   └── DIM256/
├── data/
│   ├── raw/       # Downloaded books
│   └── processed/ # Cleaned and tokenized data
├── python/        # Python scripts (download and preprocessing)
└── cuda/          # GPU version + Colab notebook
```

---

## Quick Start

Compile:
```bash
make
```
Run:
```bash
make run
```
Test:
```bash
make test
```

---

## Dataset

Download books from Project Gutenberg:
```bash
python3 python/download_gutenberg.py --books [NUM_BOOKS] --out data/raw/
```

Clean and preprocess:
```bash
python3 python/clean_books.py data/raw/ -o data/processed/output.txt --max-mb [SIZE_MB]
```

### Cleaning pipeline

The preprocessing script applies:

- Remove XML/HTML tags and entities
- Remove URLs and emails
- Convert to lowercase
- Replace numbers with @
- Normalize punctuation (; → ., : → ,)
- Remove unwanted symbols (keep . , ? ! ' -)
- Collapse multiple spaces
- Remove short lines (<5 words)
- Add $ as paragraph separator

---

## Model configuration

Model parameters are defined at the beginning of 'src/main.c'

| Parameter     | DIM128  | DIM256  |
|---------------|---------|---------|
| VOCAB SIZE    | 6000    | 10000   |
| MODEL DIM     | 128     | 256     |
| HEADS         | 4       | 4       |
| LAYERS        | 3       | 4       |
| EPOCHS        | 120     | 100     |
| WARMUP        | 5       | 4       |
| LAMBDA        | 0.0001  | 0.001   |
| MIN LR        | 0.00001 | 0.00001 |
| MAX LR        | 0.0005  | 0.0005  |
| BATCH SIZE    | 32      | 32      |
| DROPOUT       | 0.1     | 0.1     |

**Changing these parameters requires recompilation**

---

## Training performance

- Dimension = 128: 43s/MB (CPU)
- Dimension = 256: 180s/MB (CPU)

Example:
A 10MB dataset with dimension 128 -> 43 * 10 = 430s = 7min 10s (CPU)

These results were obtained on a Ryzen 5 6600H CPU.

---

## GPU training

The cuda/ folder contains:
- A CUDA compatible implementation.
- A Google Colab notebook in case you don't have a GPU.

Using a T4 GPU:
- 15min instead of 5 hours per epoch.

The colab notebook includes a cell for compiling. If you have a GPU use:
```bash
nvcc -O3 -use_fast_math -Icuda cuda/nn.cu cuda/main.cu -lcublas -lm -o build/nn_cuda 2>&1
```
And run:
```bash
./build/nn_cuda models/DIM256/pesos.bin data/processed/DIM256/token_books.bin
```
For testing, use the normal CPU version with:
```bash
make test
```

Google Colab disconnects you from the execution environment when you reach the session limits. The notebook automatically saves the data when the training is complete.

There is no version for CUDA of the tokenizer, so remember to upload the tokenized dataset to Colab. 'vocab' and 'merges' files are not needed. The notebook is easy to read and indicates every step.

---

## Tokenization

Running:
```bash
make run
```
Will:
- Generate tokenizer files if they don't exist.
- Skip regeneration if already present.

---

## Testing

There is a testing file, run it with:
```bash
make test
```

At the beginning of 'src/test.c' you can modify the temperature:
- If output is too repetitive -> increase temperature (0.9 – 1.0)
- If output is incoherent -> decrease temperature (0.6 – 0.7)

If you trained it with books, here there are some ideas for testing:
- "it was a dark and stormy night"
- "she looked over the window"
- "the old man looked at the sea"
- "she opened the door and saw"
- "in the beginning there was"

---

## Cleaning

Remove compiled binaries:
```bash
make clean
```
Remove models:
```bash
make cleandata
```

---

## Sample outputs

Dimension = 128 model, these results were achieved after 120 epochs with a 12MB dataset (30 books):

Input:
> the old man looked at the sea 

Output:
> the old man looked at the sea before him, as for the sergeant ship by starbucks!so he cant make a fine whale! 

Input:
> she looked over the window

Output:
> she looked over the window, looked round her head, and broke off to the door and saw her head with her, with the young lady caught him out, like a servant with its own eyes, looked round.

Input:
> she opened the door and saw

Output:
> she opened the door and saw this and threatened her with her early morning she had thrown back. she turned round to the hall, lay down her in crying herself with her shoulders, her eyes, squeezed her arms she raised her eyes on the pillow, looked out with a faint arm coat of cards. her chest was now that she was an awful indifference between her sister-in-law. 

Input:
> in the beginning there was

Output:
> in the beginning there was no account of the matter now, because i heard, it was not a sight of the sea.

The model learns basic grammar and sentence structure, but remains limited by its small size.

I haven't finished the training of the bigger model, i will add here some generation examples when it is completed. I am training it in Colab with the GPU version with a 100MB dataset (186 books)

---
