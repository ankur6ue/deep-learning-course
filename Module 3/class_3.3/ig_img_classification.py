
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import time
from torchvision.models import resnet50, ResNet50_Weights

import torch

# batched version
def ig_grads_batched(model, x, baseline, target, alphas, steps=200):
    """
    Compute sum of gradients along the straight-line path in a single batched pass.

    Returns:
      attr: [B,3,H,W]
    """
    model.eval()
    B = x.shape[0]
    S = alphas.shape[0]
    device, dtype = x.device, x.dtype

    # 1) Build all interpolated inputs: [S, B, ...]
    # xs = baseline + alpha * (x - baseline)
    delta = (x - baseline)
    # Make alphas broadcast to input shape
    view_shape = (S,) + (1,) * x.ndim  # (S,1,1,1,1) for images
    xs = baseline.unsqueeze(0) + alphas.view(view_shape).to(device=device, dtype=dtype) * delta.unsqueeze(0) # [S, B, C, H, W]

    # 2) Flatten to mega-batch: [S*B, ...]
    xs_flat = xs.reshape(S * B, *x.shape[1:]).clone().requires_grad_(True)

    # 3) Forward once
    out = model(xs_flat)  # [S*B, C]

    # 4) Gather fixed target logit for each example.
    # target is [B], we need it repeated S times to align with xs_flat
    target_rep = target.view(1, B).expand(S, B).reshape(S * B).to(device=device)
    logits_c = out.gather(1, target_rep.view(-1, 1)).squeeze(1)  # [S*B]

    # 5) Backward once to get grads wrt xs_flat
    # Using autograd.grad avoids touching .grad fields and is cleaner
    grads_flat = torch.autograd.grad(
        outputs=logits_c.sum(),
        inputs=xs_flat,
        create_graph=False,
        retain_graph=False
    )[0]  # [S*B, ...]

    # 6) Reshape back: [S, B, ...], sum over S -> [B, ...]
    grads = grads_flat.reshape(S, B, *x.shape[1:])
    grads_sum = grads.sum(dim=0)
    avg_grads = grads_sum / steps
    attr = (x - baseline) * avg_grads
    return attr

# -----------------------------
# Pure PyTorch Integrated Gradients (fixed target)
# -----------------------------
def integrated_gradients(
    model,
    x,                 # [B,3,H,W] normalized float
    baseline,     # same shape as x (or None -> zeros)
    target,       # int or [B] tensor; if None -> argmax at endpoint (computed once)
    alphas,
    steps=200,
):
    """
    Integrated gradients for logits (recommended for classification).
    Returns:
      attr: [B,3,H,W]
    """
    model.eval()
    x = x.detach()

    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.detach()
        if baseline.shape != x.shape:
            # allow unbatched baseline [3,H,W]
            if baseline.shape == x.shape[1:]:
                baseline = baseline.unsqueeze(0).expand_as(x)
            else:
                raise ValueError(f"baseline shape {baseline.shape} must match x shape {x.shape} (or be unbatched).")

    # Choose target ONCE at endpoint if not provided
    with torch.no_grad():
        out_x = model(x)  # logits [B,C]
        if out_x.ndim != 2:
            raise ValueError("This IG helper expects classification logits of shape [B,C].")
        if target is None:
            target = out_x.argmax(dim=1)  # [B]
        elif isinstance(target, int):
            target = torch.full((out_x.shape[0],), target, device=x.device, dtype=torch.long)
        else:
            target = target.to(x.device)

    # Riemann sum of gradients along straight-line path
    grads_sum = torch.zeros_like(x)

    for a in alphas:
        xs = (baseline + a * (x - baseline)).clone().requires_grad_(True)
        out = model(xs)  # [B,C]
        logits_c = out.gather(1, target.view(-1, 1)).squeeze(1)  # [B]

        model.zero_grad(set_to_none=True)
        if xs.grad is not None:
            xs.grad.zero_()
        logits_c.sum().backward()
        grads_sum += xs.grad.detach()

    avg_grads = grads_sum / steps
    attr = (x - baseline) * avg_grads

    # Completeness check target: sum(attr) ≈ logit_c(x) - logit_c(baseline)
    with torch.no_grad():
        out_b = model(baseline)
        logit_x = out_x.gather(1, target.view(-1, 1)).squeeze(1)
        logit_b = out_b.gather(1, target.view(-1, 1)).squeeze(1)
        delta = logit_x - logit_b

    return attr, delta, target


def input_gradients(model, x, target):
    """Raw gradients of the target logit w.r.t. input x (saliency)."""
    model.eval()
    xs = x.detach().clone().requires_grad_(True)
    out = model(xs)  # [B,C]
    logits_c = out.gather(1, target.view(-1, 1)).squeeze(1)
    model.zero_grad(set_to_none=True)
    logits_c.sum().backward()
    return xs.grad.detach()


# -----------------------------
# Visualization helpers
# -----------------------------
def download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def to_numpy_img(pil_img: Image.Image) -> np.ndarray:
    # HWC float in [0,1]
    return np.asarray(pil_img).astype(np.float32) / 255.0


def normalize_attr(attr_chw: torch.Tensor) -> np.ndarray:
    """
    attr_chw: [1,3,H,W] -> HW in [0,1]
    using abs-sum over channels.
    """
    a = attr_chw.detach().cpu().numpy()[0]   # CHW
    a = np.abs(a).sum(axis=0)                # HW
    a = a - a.min()
    a = a / (a.max() + 1e-8)
    return a

# stretch the pixel values range from percentile(lo) to percentile(hi) to cover the 0-1. Out of range values are
# clipped. This helps brighten up the integrated gradient map
def normalize_attr_percentile(attr_chw, lo=60, hi=99.5):
    a = attr_chw.detach().cpu().numpy()[0]      # BCHW -> CHW
    a = np.abs(a).sum(axis=0)                   # HW, sum along channels
    vmin = np.percentile(a, lo)
    vmax = np.percentile(a, hi)
    a = np.clip((a - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return a

def mask_overlay_bright(original_hwc, mask_hw, gamma=0.6, floor=0.25):
    """
    floor sets minimum brightness (0..1). gamma<1 brightens midrange.
    """
    m = np.power(mask_hw, gamma)
    m = floor + (1 - floor) * m                 # lift the mask away from 0
    # m[..., None] adds a third dimension to m, so m can broadcast across the 3 color channels
    return np.clip(original_hwc * m[..., None], 0, 1)


def mask_overlay(original_hwc: np.ndarray, attr_hw: np.ndarray, gamma=0.8) -> np.ndarray:
    """
    Darken everything except high-attribution regions:
      overlay = original * (attr^gamma)
    """
    m = np.power(attr_hw, gamma)[..., None]  # HWC
    return np.clip(original_hwc * m, 0, 1)

def _sync(x):
    if x.is_cuda:
        torch.cuda.synchronize()

# -----------------------------
# Main demo (matches your screenshot layout)
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pretrained ImageNet model + preprocess + class labels
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()
    labels = weights.meta["categories"]

    examples = [
        ("goose", "../data/img/swan.png"),
       # ("indigo", "../data/img/indigo_bunting.jpeg"),
        ("chameleon", "../data/img/American_chameleon.JPEG"),
        ("spider", "../data/img/garden_spider.JPEG")
    ]

    fig, axes = plt.subplots(nrows=len(examples), ncols=3, figsize=(11, 8))
    axes = np.atleast_2d(axes)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    for row, (name, path) in enumerate(examples):
        pil = Image.open(path)
        new_size = (224, 224)
        pil_resized = pil.resize(new_size)
        orig_resized = to_numpy_img(pil_resized)  # HWC

        x = preprocess(pil).unsqueeze(0).to(device)  # [1,3,H,W] normalized

        # Endpoint prediction (for display + fixed target selection)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            top_idx = int(probs.argmax(dim=1).item())
            top_label = labels[top_idx]
            top_score = float(probs[0, top_idx].item())

        baseline = torch.zeros_like(x)  # black baseline in normalized space
        # Alphas from 0->1 excluding 0 (baseline) to avoid potential gradient quirks there
        steps = 200
        target = torch.tensor([top_idx], device=device, dtype=torch.long)  # assumes B==1
        alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device, dtype=x.dtype)[1:]

        # Integrated Gradients (FIXED target = endpoint argmax)
        t0 = time.perf_counter()
        attr_ig = integrated_gradients(
            model, x, baseline=baseline, target=target, alphas=alphas, steps=200
        )
        _sync(x)
        t1 = time.perf_counter()
        t2 = time.perf_counter()
        attr_ig_batched = ig_grads_batched(
            model, x, baseline=baseline, target=target, alphas=alphas, steps=200
        )
        _sync(x)
        t3 = time.perf_counter()

        print(f"integrated_gradients time: {(t1 - t0) * 1000:.2f} ms")
        print(f"ig_grads_batched time     : {(t3 - t2) * 1000:.2f} ms")
        print(f"speedup (seq/batched)     : {(t1 - t0) / (t3 - t2):.2f}x")
        # Raw gradients at endpoint
        attr_grad = input_gradients(model, x, target=target)

        # ig_overlay = mask_overlay(orig_resized, ig_hw, gamma=0.8)
        ig_hw = normalize_attr_percentile(attr_ig_batched, lo=20, hi=99.5)
        ig_overlay = mask_overlay_bright(orig_resized, ig_hw, gamma=0.6, floor=0.25)
        grad_hw = normalize_attr_percentile(attr_grad, lo=20, hi=99.5)
        grad_overlay = mask_overlay_bright(orig_resized, grad_hw, gamma=0.6, floor=0.25)

        ax0, ax1, ax2 = axes[row]

        ax0.imshow(orig_resized)
        ax0.set_title("Original image", fontsize=10)
        ax0.axis("off")
        ax0.text(
            0.5, -0.08,
            f"Top label: {top_label}\nScore: {top_score:.6f}\n",
            transform=ax0.transAxes,
            va="top", ha="center", fontsize=9
        )

        ax1.imshow(ig_hw)
        ax1.set_title("Integrated gradients", fontsize=10)
        ax1.axis("off")

        ax2.imshow(grad_hw)
        ax2.set_title("Gradients at image", fontsize=10)
        ax2.axis("off")

    plt.suptitle("Integrated Gradients vs Input Gradients", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

