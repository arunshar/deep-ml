import torch

class MyTransform:
    def __init__(self, mean: float = 0.1307, std: float = 0.3081):
        """
        Robust init:
        - MyTransform() -> uses defaults
        - MyTransform(0.5) -> mean=0.5, std=default
        - MyTransform((0.1307, 0.3081)) -> mean/std from tuple
        """
        # If someone passes a tuple/list like (mean, std) as the first arg
        if isinstance(mean, (tuple, list)):
            if len(mean) != 2:
                raise ValueError("If passing a tuple/list, it must be (mean, std).")
            mean, std = mean  # safe unpack

        self.mean = float(mean)
        self.std = float(std)

        # Avoid divide-by-zero surprises
        if self.std == 0.0:
            self.std = 1.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, 28, 28) float tensor in [0,1]
        Return: transformed tensor, same shape/dtype.
        Must be non-identity and deterministic.
        """
        # TODO: implement your custom transformation logic here
        return (x - self.mean) / self.std