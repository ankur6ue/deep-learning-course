from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class SimpleTokenizer:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "SimpleTokenizer":
        """Build a tiny whitespace vocabulary from example texts.

        Args:
            texts: Example strings used to build the vocabulary. Tokens are
                discovered by `text.split()`, so "hello world" contributes the
                tokens "hello" and "world". Special tokens are always inserted
                first.
        """
        vocab = ["<pad>", "<bos>", "<eos>"]
        seen = set(vocab)
        for text in texts:
            for token in text.strip().split():
                if token not in seen:
                    vocab.append(token)
                    seen.add(token)
        token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        id_to_token = {idx: tok for tok, idx in token_to_id.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    @property
    def pad_token_id(self) -> int:
        """Return the integer id used for padding shorter batch rows."""
        return self.token_to_id[self.pad_token]

    @property
    def bos_token_id(self) -> int:
        """Return the integer id used to mark the start of a prompt."""
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        """Return the integer id used to mark end-of-sequence."""
        return self.token_to_id[self.eos_token]

    @property
    def vocab_size(self) -> int:
        """Return the size of the discovered vocabulary including specials."""
        return len(self.token_to_id)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        """Convert one string into token ids.

        Args:
            text: Input string split on whitespace.
            add_bos: If true, prepend the BOS token so prompts look like a
                decoder-only model input.
            add_eos: If true, append EOS. The demo usually leaves this false so
                generation can continue past the prompt.
        """
        ids = [self.token_to_id[token] for token in text.strip().split()]
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert token ids back into a space-joined string.

        Args:
            ids: Token ids to render.
            skip_special_tokens: If true, hide `<pad>`, `<bos>`, and `<eos>` so
                generated text reads more naturally.
        """
        pieces: list[str] = []
        special = {self.pad_token, self.bos_token, self.eos_token}
        for idx in ids:
            token = self.id_to_token.get(idx, "<unk>")
            if skip_special_tokens and token in special:
                continue
            pieces.append(token)
        return " ".join(pieces)
