# Coders in PyTorch

Small, beginner-friendly PyTorch examples. Every script is commented line by line to explain what each step does and why.

## Examples

| Script | What it shows |
| --- | --- |
| `01. Sequencial.py` | Trains a single-neuron model (`nn.Sequential(nn.Linear(1, 1))`) with MSE loss and SGD to learn `y = 2x - 1`, then predicts `x = 10` (≈ 19). |
| `02. CustomTokenizer.py` | Builds a word-level tokenizer by hand: lowercases and splits text, then gives each unique word an ID starting at 1 (0 is kept for padding). |
| `03. PretrainedTokenizer.py` | Uses Hugging Face's pretrained `bert-base-uncased` tokenizer: 30,522-token vocabulary, `[CLS]`/`[SEP]` special tokens, padding, and the `input_ids` / `token_type_ids` / `attention_mask` outputs. |

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch transformers numpy
```

## Running

The filenames contain spaces, so wrap them in quotes:

```bash
python "01. Sequencial.py"
python "02. CustomTokenizer.py"
python "03. PretrainedTokenizer.py"
```

`03. PretrainedTokenizer.py` downloads the BERT tokenizer from the Hugging Face Hub the first time it runs (it is cached afterwards). A warning about unauthenticated requests is harmless; set `HF_TOKEN` to remove it.

## Tested with

Python 3.13, PyTorch 2.14, Transformers 5.17, NumPy 2.5.
