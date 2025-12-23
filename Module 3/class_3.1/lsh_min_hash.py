# Copyright 2025 Ankur Mohan
import random
import hashlib
from collections import defaultdict

# -------------------------
# Utilities
# -------------------------

def tokenize(text):
    return set(text.lower().split())

def hash_fn(a, b, p):
    return lambda x: (a * x + b) % p

# -------------------------
# MinHash
# -------------------------

class MinHasher:
    def __init__(self, num_hashes=100, max_token_id=10_000):
        self.p = 100_003  # prime
        rng = random.Random(42)
        self.hashes = [
            hash_fn(
                # first coefficient a must not be 0, otherwise the hash function won't depend on x, the data point
                rng.randint(1, self.p - 1),
                rng.randint(0, self.p - 1),
                self.p,
            )
            for _ in range(num_hashes)
        ]

    def signature(self, token_ids):
        sig = []
        for h in self.hashes:
            sig.append(min(h(t) for t in token_ids))
        return sig

# -------------------------
# LSH
# -------------------------

class LSH:
    def __init__(self, bands, rows):
        self.bands = bands
        self.rows = rows
        self.tables = [defaultdict(list) for _ in range(bands)]

    def _band_hash(self, band):
        return hashlib.md5(str(band).encode()).hexdigest()

    def insert(self, doc_id, signature):
        for i in range(self.bands):
            band = signature[i*self.rows:(i+1)*self.rows]
            h = self._band_hash(band)
            self.tables[i][h].append(doc_id)

    def query(self, signature):
        candidates = set()
        for i in range(self.bands):
            band = signature[i*self.rows:(i+1)*self.rows]
            h = self._band_hash(band)
            candidates.update(self.tables[i].get(h, []))
        return candidates

# -------------------------
# Demo
# -------------------------

docs = {
    "d1": "deep learning neural networks",
    "d2": "deep learning models",
    "d3": "pizza pasta tiramisu",
}

# Token → ID
vocab = {}
def get_id(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab) + 1
    return vocab[tok]

token_sets = {
    k: {get_id(t) for t in tokenize(v)}
    for k, v in docs.items()
}

mh = MinHasher(num_hashes=100)
sigs = {k: mh.signature(v) for k, v in token_sets.items()}

lsh = LSH(bands=20, rows=5)
for doc_id, sig in sigs.items():
    lsh.insert(doc_id, sig)

query = "deep neural models"
q_tokens = {get_id(t) for t in tokenize(query)}
q_sig = mh.signature(q_tokens)

print("LSH candidates:", lsh.query(q_sig))
A = [50, 5,6, 20]
a = 2
b = 3
for i in A:
    print((a*i + b)%13)
