# Goal: build a tiny word-level tokenizer by hand. Every unique word gets a number (index),
# because neural networks work with numbers, not text.

import torch  # PyTorch (imported but not used yet; the indices could later be turned into tensors)

# Training text: two short sentences that differ in only one word ("sunny" vs "rainy").
sentences = [
    'Today is a sunny day',  # Sentence 1
    'Today is a rainy day'   # Sentence 2
]

# Tokenization Function
# Splits raw text into tokens (here, a token is one word).
def tokenize(text):
    # .lower() makes "Today" and "today" count as the same word;
    # .split() cuts on whitespace -> ['today', 'is', 'a', 'sunny', 'day']
    return text.lower().split()

# Build the vacabulary
# The vocabulary is a dict mapping each unique word -> an integer ID.
def build_vocab(sentences):
    vocab = {}  # Start empty; it fills up as new words are seen
    for sentence in sentences:  # Go through each sentence
        tokens = tokenize(sentence)  # Turn the sentence into a list of lowercase words
        for token in tokens:  # Go through each word
            if token not in vocab:  # Only add words not seen before (no duplicates)
                # The ID is the current vocab size + 1, so IDs start at 1.
                # ID 0 is left free, usually kept for padding (filling short sentences to equal length).
                vocab[token] = len(vocab)+1
    return vocab  # e.g. {'today': 1, 'is': 2, 'a': 3, 'sunny': 4, 'day': 5, 'rainy': 6}

# Tokenize the sentences
# Build the word -> ID mapping from our two sentences.
vocab = build_vocab(sentences)

# Show the result. 'rainy' gets 6 because it is the only new word in sentence 2.
print(f"Vocabulary Index: {vocab}")
