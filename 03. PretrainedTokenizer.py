# Goal: do the same job as 02. CustomTokenizer.py, but with BERT's ready-made (pretrained) tokenizer
# from Hugging Face instead of building the vocabulary ourselves.

from transformers import BertTokenizer  # Hugging Face class that loads and runs BERT's tokenizer

# Same two example sentences as in the custom tokenizer script.
sentences = [
    'Today is a sunny day',  # Sentence 1
    'Today is a rainy day'   # Sentence 2
]

# Download (the first time; cached afterwards) the tokenizer used by 'bert-base-uncased'.
# "uncased" means it lowercases text, just like our custom tokenize() did.
# Its vocabulary is fixed and was built from a huge text corpus, not from our sentences.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Number of tokens BERT knows: 30522 (whole words, word pieces like "##ing", and special tokens).
print(f"Vocabulary: {tokenizer.vocab_size}")

# Turn the sentences into model-ready numbers:
#   padding=True       -> pad shorter sentences with [PAD] (ID 0) so all rows have the same length
#   truncation=True    -> cut sentences longer than the model's max length (512 tokens for BERT)
#   return_tensors='pt'-> return PyTorch tensors instead of Python lists
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# The result is a dict-like object with three tensors:
#   input_ids      -> token IDs, with [CLS] (101) added at the start and [SEP] (102) at the end
#   token_type_ids -> which sentence each token belongs to (all 0 here, because each input is one sentence)
#   attention_mask -> 1 = real token, 0 = padding the model should ignore
print(f"Encoded Sentences: {encoded_inputs}")

# Change each row of IDs back into readable tokens, to see what the tokenizer produced,
# e.g. ['[CLS]', 'today', 'is', 'a', 'sunny', 'day', '[SEP]']
tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in encoded_inputs["input_ids"]]

# Get the whole vocabulary as a dict: token -> ID (like our custom `vocab`, but 30522 entries).
word_index = tokenizer.get_vocab()
print("Tokens: ",tokens)  # The token strings for each sentence
print("Token IDs:", encoded_inputs["input_ids"])  # The matching numeric IDs (a 2 x 7 tensor)
# Show only the first 10 vocabulary entries, because printing all 30522 would flood the screen.
# Note: get_vocab() is not sorted by ID, so these 10 are effectively random words
# (e.g. 'garden': 3871). Use sorted(word_index.items(), key=lambda kv: kv[1])[:10] to see IDs 0-9.
print("Word Index: ", dict(list(word_index.items())[:10]))
