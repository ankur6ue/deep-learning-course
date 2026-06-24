from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


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


@dataclass
class HFTokenizer:
    """Thin wrapper around a Hugging Face tokenizer.

    The teaching engine only needs a small tokenizer surface area: encode,
    decode, and access to pad/BOS/EOS ids. This wrapper keeps the engine code
    independent of the full Hugging Face tokenizer API.
    """

    backend: Any

    @classmethod
    def from_pretrained(cls, model_path: str) -> "HFTokenizer":
        """Load a tokenizer from a local Hugging Face model directory.

        Args:
            model_path: Directory containing `tokenizer.json` and related files.
        """
        from transformers import AutoTokenizer

        try:
            backend = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
        except TypeError:
            backend = AutoTokenizer.from_pretrained(model_path)
        return cls(backend=backend)

    @property
    def pad_token_id(self) -> int:
        """Return the id used to pad batch rows.

        Many decoder-only tokenizers do not define a dedicated pad token. In
        that case we reuse EOS for padding, which is a common practical choice
        for inference batching.
        """
        if self.backend.pad_token_id is not None:
            return int(self.backend.pad_token_id)
        if self.backend.eos_token_id is None:
            raise ValueError("Tokenizer does not define either pad_token_id or eos_token_id")
        return int(self.backend.eos_token_id)

    @property
    def bos_token_id(self) -> int | None:
        """Return the BOS token id if the tokenizer defines one."""
        if self.backend.bos_token_id is None:
            return None
        return int(self.backend.bos_token_id)

    @property
    def eos_token_id(self) -> int:
        """Return the EOS token id."""
        if self.backend.eos_token_id is None:
            raise ValueError("Tokenizer does not define eos_token_id")
        return int(self.backend.eos_token_id)

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        return int(len(self.backend))

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        """Encode one string into token ids.

        Args:
            text: Input prompt text.
            add_bos: If true and the tokenizer defines BOS, prepend it.
            add_eos: If true, append EOS.
        """
        ids = list(self.backend.encode(text, add_special_tokens=False))
        if add_bos and self.bos_token_id is not None:
            if not ids or ids[0] != self.bos_token_id:
                ids = [self.bos_token_id] + ids
        if add_eos and (not ids or ids[-1] != self.eos_token_id):
            ids = ids + [self.eos_token_id]
        return ids

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        """Apply the tokenizer's chat template and return token ids.

        Args:
            messages: Chat messages such as
                `[{"role": "user", "content": "Explain attention"}]`.
        """
        if not getattr(self.backend, "chat_template", None):
            text_parts = []
            for message in messages:
                role = str(message.get("role", "user")).strip() or "user"
                content = str(message.get("content", ""))
                text_parts.append(f"{role}: {content}")
            text_parts.append("assistant:")
            return self.encode("\n\n".join(text_parts), add_bos=True, add_eos=False)
        if not hasattr(self.backend, "apply_chat_template"):
            raise ValueError("Tokenizer does not provide apply_chat_template")
        encoded = self.backend.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids
            if input_ids and isinstance(input_ids[0], list):
                return list(input_ids[0])
            return list(input_ids)
        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
            if input_ids and isinstance(input_ids[0], list):
                return list(input_ids[0])
            return list(input_ids)
        return list(encoded)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids back into text."""
        return str(self.backend.decode(ids, skip_special_tokens=skip_special_tokens))
