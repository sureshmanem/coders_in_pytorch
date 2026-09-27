# Plan: Create Our Own Word Embedding

The next step after `02. CustomTokenizer.py` (words → IDs) and `03. PretrainedTokenizer.py` (BERT's IDs).
Goal: turn each word ID into a **learned vector** in which words with similar meanings end up close together.

---

## 1. Why IDs are not enough

| Representation | Example for "sunny" | Problem |
| --- | --- | --- |
| Token ID | `4` | The number means nothing: 4 is not "closer" to 5 than to 100. |
| One-hot | `[0,0,0,0,1,0,...]` | As long as the vocabulary (30,522 for BERT), and every word is equally far from every other. |
| **Embedding** | `[0.21, -0.73, 0.05, ...]` (e.g. 16–300 numbers) | Short, and **learned**: "sunny" ends up near "warm" and "bright". |

An embedding is just a lookup table: a matrix of shape `(vocab_size, embedding_dim)`.
Row `i` is the vector for word ID `i`. In PyTorch this is `nn.Embedding`, and its values
start random and are learned with the same loop as `01. Sequencial.py`
(`zero_grad → forward → loss → backward → step`).

```python
emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=16, padding_idx=0)
vectors = emb(torch.tensor([1, 4]))   # shape (2, 16): one vector per ID
```

---

## 2. Roadmap

| Step | File | What we build |
| --- | --- | --- |
| 1 | `04. EmbeddingBasics.py` | Explore `nn.Embedding`: shapes, padding, lookups (no training). |
| 2 | `05. Word2VecSkipGram.py` | **Train our own embedding from raw text** (main goal). |
| 3 | `06. EmbeddingEvaluation.py` | Check quality: similar words, word analogies, 2-D plot. |
| 4 | `07. EmbeddingForClassification.py` | Reuse the trained embedding in a small sentiment classifier. |

---

## 3. Step 1: Embedding basics (`04. EmbeddingBasics.py`)

- Reuse `tokenize()` and `build_vocab()` from `02. CustomTokenizer.py`, but reserve two special tokens:
  - `<pad>` = 0: fills short sentences so a batch has equal lengths (`padding_idx=0` keeps its vector at zero and never updates it).
  - `<unk>` = 1: any word not in the vocabulary.
- Turn sentences into padded ID tensors (shape `(batch, seq_len)`).
- Pass them through `nn.Embedding` and print the shape: `(batch, seq_len, embedding_dim)`.
- Show that before training, the similarity between words is random.

---

## 4. Step 2: Train our own embedding with Word2Vec Skip-Gram (`05. Word2VecSkipGram.py`)

**Idea:** "You shall know a word by the company it keeps." Words that appear in similar
contexts should get similar vectors. No labels are needed; the text itself is the training data.

### 4.1 Data
- Start with a small hand-written corpus (weather sentences expanding on the ones in `02`/`03`),
  then move to a real text file (e.g. a public-domain book, or the `wikitext-2` dataset).
- Lowercase + split (same as `tokenize()`); later, optionally strip punctuation.
- Drop very rare words (e.g. count < 2) → map them to `<unk>`.

### 4.2 Training pairs (window = 2)
For every word (the *center*), each neighbor within 2 positions is a *context* word:

```
"today is a sunny day"
center = "a"  →  (a, today), (a, is), (a, sunny), (a, day)
```

### 4.3 Model: two embedding tables + negative sampling
- `in_embed`: the vectors we keep at the end (the "word embedding").
- `out_embed`: helper vectors used only during training.
- Score of a pair = dot product of the two vectors. Train so that:
  - real (center, context) pairs → high score,
  - K = 5 random "negative" words → low score.

```python
class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.in_embed  = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.out_embed = nn.Embedding(vocab_size, dim, padding_idx=0)

    def forward(self, center, context, negatives):
        v   = self.in_embed(center)                                  # (B, D)
        pos = (v * self.out_embed(context)).sum(-1)                   # (B,)
        neg = torch.bmm(self.out_embed(negatives), v.unsqueeze(2)).squeeze(2)  # (B, K)
        return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(1)).mean()
```

### 4.4 Training loop
- Optimizer: `Adam(lr=0.01)` (SGD from `01` also works, just more slowly).
- Shuffle the pairs every epoch and use batches of 64–512.
- Negatives: `torch.randint(2, vocab_size, (batch, K))`, skipping `<pad>` and `<unk>`.
  (Upgrade later: sample by word frequency^0.75, as the original Word2Vec does.)
- Print the loss every epoch; it should go down and then level off.

### 4.5 Starting hyperparameters

| Setting | Tiny corpus | Real text |
| --- | --- | --- |
| `embedding_dim` | 16 | 100–300 |
| window | 2 | 5 |
| negatives K | 5 | 5–15 |
| epochs | 30 | 3–10 |
| min word count | 1 | 5 |

A quick trial on 8 weather sentences (dim 16, 30 epochs) already gave
`sunny → sun, warm` and `rainy → wet`.

---

## 5. Step 3: Evaluate (`06. EmbeddingEvaluation.py`)

1. **Nearest neighbors:** normalize the vectors, then use cosine similarity `E @ E[word]` and take the top-k.
2. **Analogies** (only meaningful on large text): `king - man + woman ≈ queen`.
3. **Plot:** reduce the vectors to 2-D with PCA (`torch.pca_lowrank`) or t-SNE and draw them with matplotlib;
   related words should form clusters.
4. **Sanity checks:** compare against the untrained (random) embedding; the trained one should look clearly better.

---

## 6. Step 4: Use the embedding (`07. EmbeddingForClassification.py`)

- Save: `torch.save({"vocab": vocab, "weights": model.in_embed.weight.data}, "my_embedding.pt")`
  (`*.pt` is already in `.gitignore`).
- Load into a new model: `nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)`.
- Small classifier: embedding → average the word vectors → `nn.Linear(dim, 2)` → positive/negative.
- Compare 3 versions: random embedding / our frozen embedding / our embedding fine-tuned (`freeze=False`).

---

## 7. Alternatives (for later)

| Approach | How it differs |
| --- | --- |
| CBOW | Predicts the center word from its context (reverse of skip-gram); faster, slightly worse for rare words. |
| Task-trained embedding | Skip Word2Vec: train `nn.Embedding` directly inside the classifier. Simple, but needs labeled data. |
| GloVe | Learns from a word co-occurrence count matrix instead of sliding windows. |
| Contextual (BERT) | The same word gets a different vector in each sentence; see `03. PretrainedTokenizer.py` → `BertModel`. |

---

## 8. Pitfalls to watch

- **Tiny corpus = noisy results.** Neighbors will look partly random until there are thousands of sentences.
- Very common words ("is", "a", "the") dominate the pairs. Fix: sub-sample frequent words or remove stop-words.
- Always use `padding_idx=0` so padding doesn't learn a meaning.
- Set `torch.manual_seed(...)` so runs can be reproduced.
- The vocabulary must be saved **together with** the weights; without it the ID → word mapping is lost.

---

## 9. Checklist

- [ ] `04. EmbeddingBasics.py`: padding, `<unk>`, shapes
- [ ] `05. Word2VecSkipGram.py`: pairs, model, training loop, save
- [ ] `06. EmbeddingEvaluation.py`: neighbors, PCA plot
- [ ] `07. EmbeddingForClassification.py`: load, freeze vs fine-tune
- [ ] Add each new script to the README table
