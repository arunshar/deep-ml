import numpy as np

class NeuralNetwork:
    """
    Build a neural network from scratch using only NumPy.
    Architecture: 784 → 128 (ReLU) → 10 (Softmax)
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01, seed=42):
        """
        Initialize network parameters using He initialization for ReLU.
        """
        self.lr = lr
        rng = np.random.default_rng(seed)

        # He init for ReLU layer: std = sqrt(2 / fan_in)
        self.W1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size), dtype=np.float32)

        # For the output layer (linear -> softmax), Xavier is also fine; He works OK too.
        # We'll use Xavier here: std = sqrt(1 / fan_in)
        self.W2 = rng.standard_normal((hidden_size, output_size)) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size), dtype=np.float32)

        # cache for forward intermediates
        self.cache = {}

    @staticmethod
    def _relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def _softmax(logits):
        """
        Numerically stable softmax.
        logits: (N, C)
        """
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X: Input batch, shape (N, 784)

        Returns:
            probs: Class probabilities, shape (N, 10)
        """
        # Layer 1: linear
        z1 = X @ self.W1 + self.b1            # (N, 128)
        h1 = self._relu(z1)                   # (N, 128)

        # Layer 2: linear
        z2 = h1 @ self.W2 + self.b2           # (N, 10)

        # Softmax
        probs = self._softmax(z2)             # (N, 10)

        # Cache for backward
        self.cache = {"X": X, "z1": z1, "h1": h1, "z2": z2, "probs": probs}
        return probs

    def backward(self, X, y, probs):
        """
        Backward pass: compute gradients and update parameters.

        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,) integer class indices
            probs: Predicted probabilities, shape (N, 10)

        Returns:
            loss: Scalar cross-entropy loss (float)
        """
        N = X.shape[0]
        eps = 1e-12  # to avoid log(0)

        # ---- 1) Cross-entropy loss ----
        # probs[np.arange(N), y] gives probability assigned to the correct class
        correct_probs = probs[np.arange(N), y]
        loss = -np.mean(np.log(correct_probs + eps))

        # ---- 2) Gradient w.r.t. logits z2 (softmax + cross-entropy simplification) ----
        # dL/dz2 = (probs - one_hot(y)) / N
        dz2 = probs.copy()
        dz2[np.arange(N), y] -= 1.0
        dz2 /= N  # average over batch

        # ---- 3) Backprop through layer 2 (h1 @ W2 + b2) ----
        h1 = self.cache["h1"]  # (N, 128)

        dW2 = h1.T @ dz2                       # (128, 10)
        db2 = np.sum(dz2, axis=0, keepdims=True)  # (1, 10)
        dh1 = dz2 @ self.W2.T                  # (N, 128)

        # ---- 4) Backprop through ReLU ----
        z1 = self.cache["z1"]                  # (N, 128)
        dz1 = dh1 * (z1 > 0)                   # (N, 128)

        # ---- 5) Backprop through layer 1 (X @ W1 + b1) ----
        dW1 = X.T @ dz1                        # (784, 128)
        db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, 128)

        # ---- 6) Gradient descent parameter update ----
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return float(loss)

    def train_step(self, X, y):
        """
        Complete training step: forward + backward + update.
        """
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss

    def predict(self, X):
        """
        Predict class labels.
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)