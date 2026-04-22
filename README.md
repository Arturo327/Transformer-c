# Transformer in C

A lightweight transformer implementation written from scratch in C, with optional CUDA acceleration for GPU training.

---

## Characteristics
- Pure C implementation (no frameworks)
- Byte Pair Encoding (BPE) Tokenizer
- Training on custom datasets (Project Gutenberg)
- CPU and CUDA (GPU) versions
- Configurable model sizes

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
├── include/       # Header file
├── build/         # Compiled binaries
├── models/        # Trained weights and tokenizer files
│   ├── DIM128/   
│   └── DIM256/
├── data/
│   ├── raw/       # Downloaded books
│   └── processed/ # Cleaned and tokenized data
├── python/        # scripts (download and preprocessing)
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

Remove XML/HTML tags and entities
Remove URLs and emails
Convert to lowercase
Replace numbers with @
Normalize punctuation (; → ., : → ,)
Remove unwanted symbols (keep . , ? ! ' -)
Collapse multiple spaces
Remove short lines (<5 words)
Add $ as paragraph separator

---

## Model configuration

Model parameters are defined at the beginning of 'src/main.c'

### Dimension = 128
```c
#define VOCAB_SIZE 6000
#define MODEL_DIM 128
#define HEADS 4
#define LAYERS 3

#define LAMBDA 0.0001f
#define MIN_LR 0.00001f
#define MAX_LR 0.0005f
#define WARMUP 5

#define BATCH_SIZE 32
#define MAX_TOKEN_LEN 32
#define TEMPERATURA 0.6f
#define TOTAL_EPOCHS 120
```

### Dimension = 256
```c
#define VOCAB_SIZE 10000
#define MODEL_DIM 256
#define HEADS 4
#define LAYERS 4

#define LAMBDA 0.001f
#define MIN_LR 0.00001f
#define MAX_LR 0.0005f
#define WARMUP 4

#define BATCH_SIZE 32
#define MAX_TOKEN_LEN 32
#define TEMPERATURA 0.6f
#define TOTAL_EPOCHS 100
```

**Changing these parameters require recomplation**

---

## Training performance

- Dimension = 128: 43s/MB (CPU)
- Dimension = 256: 180s/MB (CPU)

Example:
A 10MB dataset with dimension 128 -> 43 * 10 = 430s = 7min 16s (CPU)

This data are a result of training with a Ryzen 5 6600H.

---

## GPU training

The cuda/ folder contains:
- A CUDA compatible implementation.
- A Google Colab notebook in case you don't have a GPU.

Using a T4 GPU:
- 15min instead 5h for epoch.

Google Colab disconnects you from the execution environment when you reach the session limits. The notebook automatically saves the data when the training is complete.
There is no version for CUDA of the tokenizer, so remember to upload to Colab the tokenize book. 'vocab' and 'merges' files are no needed.

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

Dimension = 128 model, these results are achieved after 120 epochs with a 12MB dataset (30 books):

- the old man looked at the sea before him, as for the sergeant ship by starbucks!so he cant make a fine whale! 
- she looked over the window, looked round her head, and broke off to the door and saw her head with her, with the young lady caught him out, like a servant with its own eyes, looked round.
- she opened the door and saw this and threatened her with her early morning she had thrown back. she turned round to the hall, lay down her in crying herself with her shoulders, her eyes, squeezed her arms she raised her eyes on the pillow, looked out with a faint arm coat of cards. her chest was now that she was an awful indifference between her sister-in-law. 
- in the beginning there was no account of the matter now, because i heard, it was not a sight of the sea.

The model learns basic grammar and sentence structure, but remains limited by its small size.

I haven't finished the training of the bigger model, i will add here some generation examples when is completed. I am training it in Colab with the GPU version with a 100MB dataset (186 books)

---
