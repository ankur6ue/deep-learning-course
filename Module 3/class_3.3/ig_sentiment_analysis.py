from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import torch
import torch.nn.functional as F


MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def aggregate_roberta_word_attributions(tokens, token_attrib):
    """
    Aggregate token-level attributions to word-level for RoBERTa BPE tokens.
    - tokens: list[str] (from tokenizer.convert_ids_to_tokens)
    - token_attrib: 1D tensor/array/list of same length (attribution per token)

    Returns:
      words: list[str]
      scores: list[float]
    """
    # Ensure python floats
    if hasattr(token_attrib, "detach"):
        scores_tok = token_attrib.detach().cpu().tolist()
    else:
        scores_tok = list(token_attrib)

    words = []
    scores = []

    cur_word = ""
    cur_score = 0.0

    def flush():
        nonlocal cur_word, cur_score
        if cur_word != "":
            words.append(cur_word)
            scores.append(cur_score)
        cur_word = ""
        cur_score = 0.0

    specials = {"<s>", "</s>", "<pad>"}

    for tok, s in zip(tokens, scores_tok):
        if tok in specials:
            # end current word on special boundary
            flush()
            continue

        # RoBERTa: "Ġ" indicates a new word starting with a space
        if tok.startswith("Ġ"):
            flush()
            piece = tok[1:]  # drop marker
            cur_word = piece
            cur_score = float(s)
        else:
            # continuation of current word (or first word if none yet)
            piece = tok
            if cur_word == "":
                cur_word = piece
                cur_score = float(s)
            else:
                cur_word += piece
                cur_score += float(s)

    flush()
    return words, scores

# -----------------------
# Integrated Gradients (embedding-level) for HF sequence classifiers
# -----------------------
def integrated_gradients_tokens(
    model,
    encoded_input,
    target_class=None,      # int or None -> predicted class at endpoint
    steps=200,
    baseline="pad",         # "pad" or "mask"
):
    """
    Returns token attributions for a single example (batch size 1).
    Attributions are computed on the class LOGIT (recommended for stability).
    """
    input_ids = encoded_input["input_ids"].to(device)            # [1, T]
    attention_mask = encoded_input.get("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)                   # [1, T]

    B, T = input_ids.shape
    assert B == 1, "This helper expects batch size 1 for simplicity."

    # Choose target class ONCE at endpoint if not provided
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [1, C]
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

    # Baseline input ids (same shape)
    if baseline == "pad":
        pad_id = model.config.pad_token_id
        if pad_id is None:
            # RoBERTa usually has pad_token_id; fallback to tokenizer
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
        baseline_ids = torch.full_like(input_ids, pad_id)
        # keep <s> (RoBERTa CLS) token the same for stability
        baseline_ids[:, 0] = input_ids[:, 0]
    elif baseline == "mask":
        mask_id = tokenizer.mask_token_id
        if mask_id is None:
            raise ValueError("Tokenizer has no [MASK] token; use baseline='pad'.")
        baseline_ids = input_ids.clone()
        # mask everything except special tokens
        # (for RoBERTa: <s> ... </s>)
        baseline_ids[:, 1:-1] = mask_id
    else:
        raise ValueError("baseline must be 'pad' or 'mask'")

    emb_layer = model.get_input_embeddings()
    x_emb = emb_layer(input_ids)          # [1, T, d]
    b_emb = emb_layer(baseline_ids)       # [1, T, d]
    delta = x_emb - b_emb

    # Alphas: [S]
    alphas = torch.linspace(0, 1, steps + 1, device=device, dtype=x_emb.dtype)[1:]  # exclude 0
    S = alphas.shape[0]
    d = x_emb.shape[-1]

    # Build all interpolated embeddings: [S, 1, T, d] -> flatten to [S, T, d]
    xs = b_emb.unsqueeze(0) + alphas.view(S, 1, 1, 1) * delta.unsqueeze(0)    # [S,1,T,d]
    xs_flat = xs.reshape(S, T, d).clone().requires_grad_(True)                 # [S,T,d]

    # Repeat attention mask to match S
    attn_rep = attention_mask.expand(S, T)                                      # [S,T]

    # Forward with inputs_embeds
    out = model(inputs_embeds=xs_flat, attention_mask=attn_rep).logits          # [S, C]
    logits_c = out[:, target_class]                                             # [S]

    # One backward to get gradients wrt embeddings
    grads = torch.autograd.grad(logits_c.sum(), xs_flat)[0]                     # [S,T,d]

    # Sum across steps, average, scale
    avg_grads = grads.mean(dim=0, keepdim=True)                                 # [1,T,d]
    attrib_emb = delta * avg_grads                                              # [1,T,d]

    # Token-level attribution: sum over embedding dim
    token_attrib = attrib_emb.sum(dim=-1).squeeze(0)                            # [T]

    return token_attrib.detach().cpu(), target_class


text = "The flight delays ruined our vacation and caused us much stress"
text = "we thoroughly enjoyed the wonderful food at the restaurant"
encoded_input = tokenizer(text, return_tensors="pt")

# Standard prediction (your code)
with torch.no_grad():
    output = model(**{k: v.to(device) for k, v in encoded_input.items()})
scores = output.logits[0].detach().cpu().numpy()
probs = softmax(scores)

ranking = np.argsort(probs)[::-1]
for i in range(probs.shape[0]):
    l = config.id2label[int(ranking[i])]
    s = probs[int(ranking[i])]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")

# IG token attributions for predicted class
token_attrib, target_class = integrated_gradients_tokens(
    model, encoded_input, target_class=None, steps=200, baseline="pad"
)

tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0].tolist())
pred_label = config.id2label[int(target_class)]
print("\nExplaining predicted class:", pred_label)

words, word_scores = aggregate_roberta_word_attributions(tokens, token_attrib)
# normalize scores:
denom = np.max(np.abs(word_scores)) + 1e-12
scores_n = word_scores / denom
print("\nWord-level IG attributions (aggregated over subword pieces):")
for w, sn in zip(words, scores_n):
    print(f"{w:>15}  {sn:+.4f}")
