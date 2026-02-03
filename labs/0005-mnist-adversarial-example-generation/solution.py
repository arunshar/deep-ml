import torch
import torch.nn as nn

def generate_adversarial_example(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module
) -> torch.Tensor:
    """
    Generate an adversarial example for input x (FGSM with a small PGD fallback).

    Guarantees:
      - ||x_adv - x||_inf <= epsilon   (via projection / clamping delta)
      - x_adv in [0, 1]               (via torch.clamp)
    Tries (best effort):
      - to fool the model: argmax(model(x_adv)) != y
    """
    device = x.device

    # Ensure y is a LongTensor of shape (1,)
    if not torch.is_tensor(y):
        y = torch.tensor([y], device=device)
    else:
        y = y.to(device)
        if y.dim() == 0:
            y = y.unsqueeze(0)
    y = y.long()

    # ---------- 1) FGSM (single step) ----------
    x_adv = x.clone().detach().requires_grad_(True)

    # Forward + loss
    logits = model(x_adv)
    loss = criterion(logits, y)

    # Backward: gradient wrt input
    model.zero_grad(set_to_none=True)
    if x_adv.grad is not None:
        x_adv.grad.zero_()
    loss.backward()

    # Take a step that *increases* loss the most (sign of gradient)
    grad_sign = x_adv.grad.sign()
    x_adv = x_adv + epsilon * grad_sign

    # Clamp to valid pixel range
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # Project back to L_inf epsilon-ball around original x
    delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
    x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    # If it already fools the model, return
    with torch.no_grad():
        if model(x_adv).argmax(dim=1).item() != y.item():
            return x_adv

    # ---------- 2) Small PGD fallback (few steps) ----------
    # This increases success rate while still respecting the same constraints.
    steps = 10
    alpha = max(epsilon / 4.0, 1e-3)  # step size

    x_adv = x.clone().detach()

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)

        logits = model(x_adv)
        loss = criterion(logits, y)

        model.zero_grad(set_to_none=True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()

        # PGD update (maximize loss)
        x_adv = x_adv + alpha * x_adv.grad.sign()

        # Project to epsilon-ball and valid range
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

        # Early stop if fooled
        with torch.no_grad():
            if model(x_adv).argmax(dim=1).item() != y.item():
                break

    return x_adv