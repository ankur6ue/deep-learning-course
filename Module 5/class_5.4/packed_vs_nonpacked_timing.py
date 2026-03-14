import time
import torch
import torch.nn.functional as F

# -----------------------
# Config
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

B = 64          # batch
T = 512         # seq length
d_model = 512
h = 8
d_head = d_model // h
assert d_head * h == d_model

dtype = torch.float16  # try float32 too
use_amp = (device == "cuda" and dtype == torch.float16)

# Input
X = torch.randn(B, T, d_model, device=device, dtype=torch.float32)

# -----------------------
# Build weights so packed & non-packed are identical
# -----------------------
# Non-packed per-head weights: [h, d_head, d_model]
Wq = torch.randn(h, d_head, d_model, device=device, dtype=torch.float32)
Wk = torch.randn(h, d_head, d_model, device=device, dtype=torch.float32)
Wv = torch.randn(h, d_head, d_model, device=device, dtype=torch.float32)

# Packed weights: [h*d_head, d_model] for each of Q,K,V
Wq_packed = Wq.reshape(h * d_head, d_model).contiguous()
Wk_packed = Wk.reshape(h * d_head, d_model).contiguous()
Wv_packed = Wv.reshape(h * d_head, d_model).contiguous()

# Optionally: one single "QKV packed" weight (common in impls): [3*h*d_head, d_model]
Wqkv_packed = torch.cat([Wq_packed, Wk_packed, Wv_packed], dim=0).contiguous()

# Output proj weight (same for both)
Wo = torch.randn(d_model, d_model, device=device, dtype=torch.float32)

# -----------------------
# Two implementations
# -----------------------
def proj_non_packed(X):
    """
    3*h matmuls (conceptually): each head has its own Wq/Wk/Wv.
    Returns Q,K,V as (B, h, T, d_head)
    """
    # Use F.linear: y = x @ W^T. Our W are shaped (d_head, d_model).
    Qs, Ks, Vs = [], [], []
    for i in range(h):
        Qi = F.linear(X, Wq[i])  # (B,T,d_head)
        Ki = F.linear(X, Wk[i])  # (B,T,d_head)
        Vi = F.linear(X, Wv[i])  # (B,T,d_head)
        Qs.append(Qi)
        Ks.append(Ki)
        Vs.append(Vi)
    Q = torch.stack(Qs, dim=1)  # (B,h,T,d_head)
    K = torch.stack(Ks, dim=1)
    V = torch.stack(Vs, dim=1)
    return Q, K, V

def proj_packed_three_matmuls(X):
    """
    3 matmuls total: Q = XWq, K = XWk, V = XWv with packed weights.
    Returns Q,K,V as (B, h, T, d_head)
    """
    Q = F.linear(X, Wq_packed)  # (B,T,h*d_head)
    K = F.linear(X, Wk_packed)
    V = F.linear(X, Wv_packed)
    Q = Q.view(B, T, h, d_head).transpose(1, 2).contiguous()  # (B,h,T,d_head)
    K = K.view(B, T, h, d_head).transpose(1, 2).contiguous()
    V = V.view(B, T, h, d_head).transpose(1, 2).contiguous()
    return Q, K, V

def proj_packed_single_matmul(X):
    """
    1 matmul for QKV total: QKV = X Wqkv^T, then split.
    Returns Q,K,V as (B, h, T, d_head)
    """
    QKV = F.linear(X, Wqkv_packed)  # (B,T,3*h*d_head)
    Q, K, V = torch.split(QKV, h * d_head, dim=-1)
    Q = Q.view(B, T, h, d_head).transpose(1, 2).contiguous()
    K = K.view(B, T, h, d_head).transpose(1, 2).contiguous()
    V = V.view(B, T, h, d_head).transpose(1, 2).contiguous()
    return Q, K, V

def attention(Q, K, V):
    """
    Standard scaled dot-product attention + output projection.
    Q,K,V: (B,h,T,d_head)
    Returns: (B,T,d_model)
    """
    scale = (d_head ** -0.5)
    # scores: (B,h,T,T)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    # out_heads: (B,h,T,d_head)
    out_heads = torch.matmul(attn, V)
    # concat: (B,T,d_model)
    out = out_heads.transpose(1, 2).contiguous().view(B, T, d_model)
    # output projection: (B,T,d_model)
    out = F.linear(out, Wo)
    return out

# -----------------------
# Correctness check (projection stage)
# -----------------------
with torch.no_grad():
    Q1, K1, V1 = proj_non_packed(X)
    Q2, K2, V2 = proj_packed_three_matmuls(X)
    Q3, K3, V3 = proj_packed_single_matmul(X)

    print("Max |Q_non - Q_packed3|:", (Q1 - Q2).abs().max().item())
    print("Max |Q_non - Q_packed1|:", (Q1 - Q3).abs().max().item())

# -----------------------
# Timing helpers
# -----------------------
def time_fn(fn, iters=200, warmup=50, label=""):
    # Use CUDA events for accurate GPU timing
    if device != "cuda":
        t0 = time.time()
        for _ in range(warmup): fn()
        t1 = time.time()
        for _ in range(iters): fn()
        t2 = time.time()
        print(f"{label:28s} warmup {t1-t0:.3f}s | iters {t2-t1:.3f}s")
        return

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    print(f"{label:28s} {ms/iters:.4f} ms/iter  ({iters} iters)")

# -----------------------
# Benchmark: projection only (shows matmul count difference clearly)
# -----------------------
def bench_proj_non():
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        proj_non_packed(X)

def bench_proj_packed3():
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        proj_packed_three_matmuls(X)

def bench_proj_packed1():
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        proj_packed_single_matmul(X)

print("\n--- Projection-only timing ---")
time_fn(bench_proj_non,    label="non-packed (3*h matmuls)")
time_fn(bench_proj_packed3,label="packed (3 matmuls)")
time_fn(bench_proj_packed1,label="packed (1 matmul)")

# -----------------------
# Benchmark: full attention (projection + QK^T + softmax + AV + Wo)
# Note: for large T, attention dominates and the projection savings shrinks.
# -----------------------
def bench_full_non():
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        Q,K,V = proj_non_packed(X)
        attention(Q,K,V)

def bench_full_packed1():
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        Q,K,V = proj_packed_single_matmul(X)
        attention(Q,K,V)

print("\n--- Full attention timing ---")
time_fn(bench_full_non,     label="full: non-packed proj")
time_fn(bench_full_packed1, label="full: packed QKV proj")