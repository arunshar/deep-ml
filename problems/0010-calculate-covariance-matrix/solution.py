def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    # 1. Determine dimensions
    n_features = len(vectors)
    n_obs = len(vectors[0])
    
    # Edge case: If there's 1 or fewer observations, covariance is undefined
    if n_obs <= 1:
        return []

    # 2. Calculate the mean for each feature
    means = [sum(feature) / n_obs for feature in vectors]
    
    # 3. Initialize an empty n x n covariance matrix
    cov_matrix = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    
    # 4. Populate the covariance matrix
    for i in range(n_features):
        for j in range(n_features):
            # Sum the product of deviations from the mean for both features
            cov_sum = sum(
                (vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) 
                for k in range(n_obs)
            )
            # Divide by (n - 1) for sample covariance
            cov_matrix[i][j] = cov_sum / (n_obs - 1)
            
    retur