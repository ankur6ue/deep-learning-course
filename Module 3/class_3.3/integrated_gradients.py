import torch

@torch.no_grad()
def _validate_baseline(x, baseline):
    if baseline is None:
        return torch.zeros_like(x)
    if baseline.shape != x.shape:
        # Allow baseline broadcast if baseline is single example
        if baseline.shape == x.shape[1:]:
            baseline = baseline.unsqueeze(0).expand_as(x)
        else:
            raise ValueError(f"baseline shape {baseline.shape} must match x {x.shape} (or be unbatched).")
    return baseline

def integrated_gradients(
    model,
    x,
    baseline=None,
    target=None,
    steps=64,
    use_logits=True,
):
    """
    Integrated Gradients for model input x.

    Args:
      model: torch.nn.Module in eval mode
      x: input tensor [B, ...] with floats; requires_grad not necessary (we set internally)
      baseline: same shape as x (or unbatched) or None -> zeros
      target: None or int or tensor [B] of target class indices (for classification)
      steps: number of interpolation steps
      use_logits: if True, explain logits; else explain probabilities (less stable)

    Returns:
      attributions: tensor same shape as x
      delta: tensor [B] approx sum(attrib) == f(x)-f(baseline)
    """
    model.eval()
    x = x.detach()
    baseline = _validate_baseline(x, baseline)

    # Generate scaled inputs along the path: [steps, B, ...]
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device, dtype=x.dtype)[1:]  # exclude 0
    scaled = baseline.unsqueeze(0) + alphas.view(-1, *([1] * x.ndim)) * (x.unsqueeze(0) - baseline.unsqueeze(0))

    # We'll accumulate gradients
    grads_sum = torch.zeros_like(x)

    for s in range(steps):
        xs = scaled[s].clone().requires_grad_(True)  # [B, ...]
        out = model(xs)

        # Choose scalar per example
        if out.ndim == 2:  # [B, C]
            if target is None:
                # default: predicted class per example
                t = out.argmax(dim=1)
            elif isinstance(target, int):
                t = torch.full((out.shape[0],), target, device=out.device, dtype=torch.long)
            else:
                t = target.to(out.device)

            if use_logits:
                scalar = out.gather(1, t.view(-1, 1)).squeeze(1)  # [B]
            else:
                probs = torch.softmax(out, dim=1)
                scalar = probs.gather(1, t.view(-1, 1)).squeeze(1)  # [B]
        else:
            # regression: out is [B] or [B,1]
            scalar = out.view(out.shape[0])

        # Backprop to inputs
        model.zero_grad(set_to_none=True)
        if xs.grad is not None:
            xs.grad.zero_()
        scalar.sum().backward()
        grads_sum += xs.grad.detach()

    avg_grads = grads_sum / steps
    attributions = (x - baseline) * avg_grads

    # Completeness check
    with torch.no_grad():
        fx = model(x)
        fb = model(baseline)
        if fx.ndim == 2 and target is not None:
            if isinstance(target, int):
                t = torch.full((fx.shape[0],), target, device=fx.device, dtype=torch.long)
            else:
                t = target.to(fx.device)
            fx = fx.gather(1, t.view(-1, 1)).squeeze(1)
            fb = fb.gather(1, t.view(-1, 1)).squeeze(1)
        else:
            fx = fx.view(fx.shape[0])
            fb = fb.view(fb.shape[0])

        delta = fx - fb  # [B]

    return attributions, delta
