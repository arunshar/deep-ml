import math
import scipy.stats as stats

def calculate_power(effect_size: float, sample_size_per_group: int, alpha: float = 0.05, two_tailed: bool = True) -> float:
    """
    Calculate statistical power for a two-sample z-test.
    """
    # 1. Calculate the Non-Centrality Parameter (NCP)
    # Formula: d * sqrt(n / 2)
    ncp = effect_size * math.sqrt(sample_size_per_group / 2.0)
    
    if two_tailed:
        # 2. Find the critical Z-value for a two-tailed test
        # We look at 1 - (alpha / 2). For alpha=0.05, this is the 97.5th percentile (~1.96)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        
        # 3. Calculate Power using the CDF
        # Power = 1 - CDF(z_crit - ncp) + CDF(-z_crit - ncp)
        power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
        
    else:
        # 2. Find the critical Z-value for a one-tailed test
        # We look at 1 - alpha. For alpha=0.05, this is the 95th percentile (~1.645)
        z_crit = stats.norm.ppf