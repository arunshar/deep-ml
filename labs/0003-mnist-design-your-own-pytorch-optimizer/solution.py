import math
import torch
from torch.optim.optimizer import Optimizer

class MyOptimizer(Optimizer):
    """
    AdamW-like optimizer (Adam with decoupled weight decay).
    - Good default for fast convergence on MNIST.
    - Only dense gradients are supported.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            # Closure is used by some optimizers / line-search methods.
            # We re-enable grad so closure can compute forward/backward.
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MyOptimizer does not support sparse gradients")

                # State for this parameter tensor
                state = self.state[p]
                if len(state) == 0:
                    # Step counter
                    state["step"] = 0
                    # Exponential moving average of gradient
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # --- AdamW: decoupled weight decay ---
                # Weight decay is applied directly to parameters, not mixed into grad.
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

                # --- Update moving averages ---
                # m_t = beta1 * m_{t-1} + (1-beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # v_t = beta2 * v_{t-1} + (1-beta2) * (g_t^2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # --- Bias correction (important early steps) ---
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t

                # step_size = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # denom = sqrt(v_t) + eps
                denom = exp_avg_sq.sqrt().add_(eps)

                # p = p - step_size * m_t / denom
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss