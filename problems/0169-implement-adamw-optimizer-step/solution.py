import numpy as np

def adamw_update(w: np.ndarray, g: np.ndarray, m: np.ndarray, v: np.ndarray, 
                 t: int, lr: float, beta1: float, beta2: float, 
                 epsilon: float, weight_decay: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one AdamW optimizer step.
    """
    # Step 1: Update moment estimates
    # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * (g ** 2)
    
    # Step 2: Bias correction
    # Compensate for the fact that m and v are initialized at zero
    m_hat = m_new / (1.0 - beta1 ** t)
    v_hat = v_new / (1.0 - beta2 ** t)
    
    # Step 3: Decoupled weight decay
    # w = w - lr * weight_decay * w
    w_decayed = w - lr * weight_decay * w
    
    # Step 4: Gradient based update
    # w = w - lr * (m_hat / (sqrt(v_hat) + epsilon))
    w_new = w_decayed - lr * (m_hat / (np.sqrt(v_hat) + epsilo