import math

def beta_distribution_stats(x: float, alpha: float, beta_param: float) -> dict:
    """
    Compute Beta distribution statistics.
    """
    # 1. Calculate the Expected Value (Mean)
    mean = alpha / (alpha + beta_param)
    
    # 2. Calculate the Variance
    variance = (alpha * beta_param) / (((alpha + beta_param) ** 2) * (alpha + beta_param + 1))
    
    # 3. Calculate the PDF
    # The problem explicitly notes that for x outside (0, 1), the PDF is 0.
    if x <= 0 or x >= 1:
        pdf = 0.0
    else:
        # Calculate the log of the Beta function using math.lgamma
        # ln(B(alpha, beta)) = ln(Gamma(alpha)) + ln(Gamma(beta)) - ln(Gamma(alpha + beta))
        log_beta_func = math.lgamma(alpha) + math.lgamma(beta_param) - math.lgamma(alpha + beta_param)
        
        # Calculate the log of the PDF
        # ln(f(x)) = (alpha - 1)*ln(x) + (beta - 1)*ln(1 - x) - ln(B(alpha, beta))
        log_pdf = (alpha - 1) * math.log(x) + (beta_param - 1) * math.log(1 - 