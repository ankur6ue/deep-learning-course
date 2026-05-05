# SPLADE from scratch: BM25 vs Dense vs SPLADE (BEIR)

This repo trains a minimal SPLADE-style sparse retriever on a BEIR dataset and compares it to:
- **BM25** (rank_bm25 baseline)
- **Dense retrieval** (SentenceTransformers + FAISS)
- **SPLADE** (this implementation)

It is designed for **teaching / reproducible experiments**, not maximum leaderboard performance.

---

## 0) Quick start (recommended: SciFact)

### Create env + install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# Core deps
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers faiss-cpu datasets tqdm numpy scikit-learn rank-bm25

# BEIR utilities + evaluation
pip install beir
```

> If you have a CUDA GPU, install the appropriate PyTorch build and set `--device cuda`.

---

## 1) Download dataset (BEIR)
```bash
python -m src.data_beir --dataset scifact --out_dir data/beir
```

This creates:
- `data/beir/<dataset>/corpus.jsonl`
- `data/beir/<dataset>/queries.jsonl`
- `data/beir/<dataset>/qrels/test.tsv` (and/or dev)

---

## 2) Run BM25 baseline
```bash
python -m src.bm25_retrieval --dataset_dir data/beir/scifact --split test --k 100 --out runs/bm25.json
python -m src.eval_beir --dataset_dir data/beir/scifact --split test --run runs/bm25.json
```

---

## 3) Run dense baseline (SentenceTransformers + FAISS)
```bash
python -m src.dense_retrieval --dataset_dir data/beir/scifact --split test --k 100 --out runs/dense.json
python -m src.eval_beir --dataset_dir data/beir/scifact --split test --run runs/dense.json
```

---

## 4) Train SPLADE (minimal)
```bash
python -m src.splade_train   --dataset_dir data/beir/scifact   --split train   --output_dir checkpoints/splade_scifact   --model bert-base-uncased   --batch_size 8   --lr 2e-5   --epochs 1   --lambda_q 1e-4   --lambda_d 1e-4   --max_len 256   --device cpu
```

Then index + retrieve:
```bash
python -m src.splade_index   --dataset_dir data/beir/scifact   --split test   --ckpt checkpoints/splade_scifact/best.pt   --out_index indices/splade_scifact_index.pkl   --doc_topk 128   --max_len 256   --device cpu

python -m src.splade_retrieval   --dataset_dir data/beir/scifact   --split test   --ckpt checkpoints/splade_scifact/best.pt   --index indices/splade_scifact_index.pkl   --k 100   --query_topk 64   --max_len 256   --device cpu   --out runs/splade.json

python -m src.eval_beir --dataset_dir data/beir/scifact --split test --run runs/splade.json
```

---

## 5) Notes / teaching hooks

### What “SPLADE” means here
We use the standard SPLADE idea: a transformer produces per-token vocabulary logits, then we form a sparse vector:
- `w[t] = max_i log(1 + relu(logit[i,t]))`

This produces non-negative sparse weights over the vocabulary; retrieval is a dot product between query and doc vectors.

### Sparsity knobs
- `--lambda_q`, `--lambda_d` control L1 penalties on query/doc vectors.
- `--doc_topk` and `--query_topk` enforce explicit top-k sparsification for indexing/retrieval.

### Compute expectations
This implementation is intentionally minimal and will not match SPLADEv2 leaderboard results. It is meant to show:
- how sparse learned expansions differ from BM25 TF counts
- how dense retrieval differs from lexical matching
- the efficiency/quality tradeoff as you vary sparsity

---

## 6) File overview

- `src/data_beir.py`: download + export BEIR dataset to JSONL/TSV
- `src/bm25_retrieval.py`: BM25 retrieval baseline
- `src/dense_retrieval.py`: SentenceTransformers + FAISS baseline
- `src/splade_model.py`: SPLADE encoder + sparse vector construction
- `src/splade_train.py`: in-batch negatives + sparsity regularization
- `src/splade_index.py`: build a Python postings-like index (top-k terms per doc)
- `src/splade_retrieval.py`: query-time scoring using the postings index
- `src/eval_beir.py`: nDCG@10 + Recall@{10,100,1000} (with BEIR eval if available)

---

## 7) Repro tips
- Start with SciFact. Then try FiQA.
- Use `--device cuda` if you have a GPU.
- Increase `--epochs`, `--batch_size` and mine better negatives if you want stronger results.

Enjoy!
