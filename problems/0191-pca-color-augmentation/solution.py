import numpy as np

def pca_color_augmentation(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    # 1. Flatten the image treating every pixel as a data point
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    # 2. Standardize: Zero-center the pixel data
    pixels_centered = pixels - np.mean(pixels, axis=0)
    
    # 3. Compute Covariance Matrix
    cov = np.cov(pixels_centered, rowvar=False)
    
    # 4. Get Eigenvalues and Eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    
    # 5. THE FIX: Sort eigenvalues and eigenvectors in DESCENDING order!
    # PCA components must be ordered from largest variance to smallest.
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    
    # 6. Calculate the shift 
    shift = np.dot(eig_vecs, alpha * np.sqrt(np.maximum(eig_vals, 0)))
    
    # 7. Add the shift to the original image and clamp
    augmented_image = image.astype(np.float32) + shift
    
    return np.clip(a