import torch

sentences = [
    'Today is a sunny day',
    'Today is a rainy day'
]

# Tokenization Function
def tokenize(text):
    return text.lower().split()

# Build the vacabulary
def build_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        tokens = tokenize(sentence)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)+1
    return vocab

# Tokenize the sentences
vocab = build_vocab(sentences)

print(f"Vocabulary Index: {vocab}")