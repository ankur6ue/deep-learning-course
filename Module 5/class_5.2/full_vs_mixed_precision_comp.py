import torch
import numpy as np

# compares gradients calculated in full vs mixed precision

def make_model(in_size, out_size, num_layers, device="cuda"):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    net = torch.nn.Sequential(*layers).to(device)
    return net

@torch.no_grad()
def clone_grads(model):
    """Return a dict: param_name -> grad tensor on CPU float64 (or None)."""
    out = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            out[name] = None
        else:
            out[name] = p.grad.detach().cpu().to(torch.float64).clone()
    return out

def grad_stats(grads_fp32, grads_amp, eps=1e-12):
    rows = []
    for name in grads_fp32.keys():
        g1 = grads_fp32[name]
        g2 = grads_amp[name]
        if g1 is None or g2 is None:
            continue

        diff = g2 - g1
        max_abs = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()

        denom = g1.abs().mean().item() + eps
        rel_mean = mean_abs / denom

        # Cosine similarity between flattened grads
        v1 = g1.reshape(-1)
        v2 = g2.reshape(-1)
        cos = torch.dot(v1, v2) / (torch.linalg.vector_norm(v1) * torch.linalg.vector_norm(v2) + eps)
        cos = cos.item()

        rows.append((name, max_abs, mean_abs, rel_mean, cos))

    # Sort by largest relative error
    rows.sort(key=lambda x: x[3], reverse=True)
    return rows

def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "This demo is intended for CUDA AMP."

    batch_size = 1024
    in_size = 4096
    out_size = 4096
    num_layers = 9

    # Use ONE batch so comparison is straightforward
    x = torch.randn(batch_size, in_size, device=device, dtype=torch.float32)
    y = torch.randn(batch_size, out_size, device=device, dtype=torch.float32)

    loss_fn = torch.nn.MSELoss().to(device)

    # ----- Full precision grads -----
    net_fp32 = make_model(in_size, out_size, num_layers, device=device)
    net_fp32.train()

    state = {k: v.detach().clone() for k, v in net_fp32.state_dict().items()}  # save identical init

    net_fp32.zero_grad(set_to_none=True)
    out = net_fp32(x)                         # fp32
    loss = loss_fn(out, y)                    # fp32
    loss.backward()
    grads_fp32 = clone_grads(net_fp32)

    # ----- AMP grads (autocast + GradScaler) -----
    net_amp = make_model(in_size, out_size, num_layers, device=device)
    net_amp.load_state_dict(state)            # same weights
    net_amp.train()

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    net_amp.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        out2 = net_amp(x)
        loss2 = loss_fn(out2, y)

    scaler.scale(loss2).backward()

    # IMPORTANT: unscale so grads are comparable (and so they’re in “real units”)
    # We can unscale without an optimizer by using a dummy optimizer, or manually divide grads.
    # Here we manually unscale using the current scale:
    scale = scaler.get_scale()
    for p in net_amp.parameters():
        if p.grad is not None:
            p.grad.detach().div_(scale)

    grads_amp = clone_grads(net_amp)

    # ----- Compare -----
    rows = grad_stats(grads_fp32, grads_amp)

    print("\nTop 10 params by relative mean abs grad diff (AMP vs FP32):")
    print("name | max_abs_diff | mean_abs_diff | rel_mean_abs_diff | cosine_sim")
    for name, max_abs, mean_abs, rel_mean, cos in rows[:10]:
        print(f"{name:30s}  {max_abs:10.3e}  {mean_abs:10.3e}  {rel_mean:10.3e}  {cos: .8f}")

    # Also print a global summary
    max_rel = rows[0][3] if rows else 0.0
    min_cos = min(r[4] for r in rows) if rows else 1.0
    print(f"\nSummary: worst rel_mean_abs_diff={max_rel:.3e}, worst cosine_sim={min_cos:.8f}")

if __name__ == "__main__":
    main()