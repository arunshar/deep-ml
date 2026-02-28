import numpy as np
from scipy import stats

def two_sample_t_test(sample1: list[float], sample2: list[float], alpha: float = 0.05) -> dict:
    """
    Perform a two-sample independent t-test (Welch's t-test).
    """
    # 1. Convert lists to numpy arrays for vectorized math
    s1 = np.array(sample1)
    s2 = np.array(sample2)
    
    n1 = len(s1)
    n2 = len(s2)
    
    # 2. Calculate means
    m1 = np.mean(s1)
    m2 = np.mean(s2)
    
    # 3. Calculate sample variances (Use ddof=1 for Bessel's correction)
    v1 = np.var(s1, ddof=1)
    v2 = np.var(s2, ddof=1)
    
    # 4. Calculate Standard Error (SE)
    se = np.sqrt((v1 / n1) + (v2 / n2))
    
    # 5. Calculate T-statistic
    t_stat = (m1 - m2) / se
    
    # 6. Calculate Degrees of Freedom (Welch-Satterthwaite equation)
    df_num = ((v1 / n1) + (v2 / n2)) ** 2
    df_den = ((v1 / n1) ** 2 / (n1 - 1)) + ((v2 / n2) ** 2 / (n2 - 1))
    df = df_num / df_den
    
    # 7. Calculate P-value (Two-tailed)
    # We use the ab