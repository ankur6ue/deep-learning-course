import re
from collections import Counter, defaultdict
from pathlib import Path


class BPETokenizer:
    def __init__(self, num_merges=200, lowercase=True, verbose=True):
        self.num_merges = num_merges
        self.lowercase = lowercase
        self.verbose = verbose

        self.merges = []               # list of merged pairs in order
        self.merge_ranks = {}          # pair -> rank
        self.vocab_counter = None      # word-piece corpus during training
        self.symbol_vocab = set()      # learned symbols

    # ----------------------------
    # Text preprocessing
    # ----------------------------
    def preprocess_text(self, text: str):
        if self.lowercase:
            text = text.lower()

        # very light normalization:
        # keep letters, digits, apostrophes; split punctuation into spaces
        text = re.sub(r"[^a-z0-9'\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def words_from_text(self, text: str):
        text = self.preprocess_text(text)
        if not text:
            return []
        return text.split()

    # ----------------------------
    # BPE training helpers
    # ----------------------------
    def word_to_symbols(self, word: str):
        # Represent a word as tuple of characters + end-of-word marker
        return tuple(list(word) + ["</w>"])

    def build_initial_vocab(self, words):
        word_freq = Counter(words)
        vocab = Counter()
        for word, freq in word_freq.items():
            vocab[self.word_to_symbols(word)] = freq
        return vocab

    def get_pair_counts(self, vocab_counter):
        pair_counts = Counter()
        for symbols, freq in vocab_counter.items():
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += freq
        return pair_counts

    def merge_pair_in_word(self, symbols, pair):
        """Merge all occurrences of pair inside one tokenized word."""
        merged = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return tuple(merged)

    def apply_merge_to_vocab(self, vocab_counter, pair):
        new_vocab = Counter()
        for symbols, freq in vocab_counter.items():
            new_symbols = self.merge_pair_in_word(symbols, pair)
            new_vocab[new_symbols] += freq
        return new_vocab

    def fit(self, text: str):
        words = self.words_from_text(text)
        if not words:
            raise ValueError("No words found in training text after preprocessing.")

        vocab = self.build_initial_vocab(words)

        if self.verbose:
            total_words = sum(vocab.values())
            print(f"Training BPE on {total_words:,} word tokens")
            print(f"Unique word forms: {len(vocab):,}")

        self.merges = []
        self.merge_ranks = {}

        for step in range(self.num_merges):
            pair_counts = self.get_pair_counts(vocab)
            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                # no repeated pair left worth merging
                break

            vocab = self.apply_merge_to_vocab(vocab, best_pair)
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = step

            if self.verbose and (step < 20 or (step + 1) % 50 == 0):
                print(f"merge {step+1:4d}: {best_pair}  count={best_count}")

        self.vocab_counter = vocab

        # build symbol vocabulary
        symbol_vocab = set()
        for symbols in vocab:
            symbol_vocab.update(symbols)
        self.symbol_vocab = symbol_vocab

        if self.verbose:
            print(f"\nLearned {len(self.merges)} merges")
            print(f"Final symbol vocabulary size: {len(self.symbol_vocab):,}")

    # ----------------------------
    # Tokenization with learned merges
    # ----------------------------
    def encode_word(self, word: str):
        symbols = list(self.word_to_symbols(word))

        # Apply merges greedily in learned order until no merge applies
        while True:
            candidate_pairs = []
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in self.merge_ranks:
                    candidate_pairs.append((self.merge_ranks[pair], i, pair))

            if not candidate_pairs:
                break

            # apply the earliest-learned applicable merge first
            _, i, pair = min(candidate_pairs)
            symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2:]

        # strip end-of-word marker if merged into final token
        out = []
        for s in symbols:
            if s == "</w>":
                continue
            if s.endswith("</w>"):
                out.append(s[:-4])
            else:
                out.append(s)
        return out

    def encode(self, text: str):
        words = self.words_from_text(text)
        return [self.encode_word(w) for w in words]

    def print_top_final_symbols(self, top_k=50):
        if self.vocab_counter is None:
            print("Tokenizer not trained yet.")
            return

        sym_counts = Counter()
        for symbols, freq in self.vocab_counter.items():
            for s in symbols:
                sym_counts[s] += freq

        print(f"\nTop {top_k} final symbols:")
        for sym, cnt in sym_counts.most_common(top_k):
            print(f"{repr(sym):20s} {cnt}")

    def print_tokenization_examples(self, texts):
        print("\nExample tokenizations:")
        for text in texts:
            pieces = self.encode(text)
            print(f"\nTEXT: {text}")
            print("TOKENS:")
            for word, tok in zip(self.words_from_text(text), pieces):
                print(f"  {word:20s} -> {tok}")


def load_corpus_from_file(path, max_chars=2_000_000):
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    if max_chars is not None:
        text = text[:max_chars]
    return text


def fallback_demo_corpus():
    # Simple fallback if you don't want to point to a dataset yet
    return """
    the food was good and the service was bad
    the food was bad and the service was good
    transformers use attention to contextualize token embeddings
    tokenization converts text into discrete units called tokens
    byte pair encoding learns frequent subword merges
    word embeddings can represent semantic similarity between words
    attention mechanisms use queries keys and values
    residual connections and layer normalization stabilize deep networks
    the restaurant service was excellent but the food was disappointing
    training neural networks requires optimization over many examples
    """ * 5000


if __name__ == "__main__":
    # ------------------------------------------------------------
    # OPTION 1:
    # Point this to the same plain-text corpus file you used for word2vec.
    # Example:
    # corpus_path = "/path/to/text8_subset.txt"
    # corpus_text = load_corpus_from_file(corpus_path, max_chars=1_000_000)
    # ------------------------------------------------------------

    corpus_path = 'data/text8'

    if corpus_path is not None:
        corpus_text = load_corpus_from_file(corpus_path, max_chars=1_000_000)
    else:
        corpus_text = fallback_demo_corpus()

    bpe = BPETokenizer(num_merges=2000, lowercase=True, verbose=True)
    bpe.fit(corpus_text)

    bpe.print_top_final_symbols(top_k=60)

    examples = [
        "the food was unbelievably good",
        "the service was disappointing",
        "attention contextualizes embeddings",
        "tokenization and transformers",
        "unbelievableness",
    ]

    bpe.print_tokenization_examples(examples)
    # Now use standard BPE that is used by gpt2
    from transformers import AutoTokenizer

    # GPT-2 uses byte-level BPE
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    for text in examples:
        enc = tokenizer(text, add_special_tokens=False)
        ids = enc["input_ids"]
        toks = tokenizer.convert_ids_to_tokens(ids)

        print(f"\nTEXT:   {text}")
        print(f"TOKENS: {toks}")
        print(f"IDS:    {ids}")