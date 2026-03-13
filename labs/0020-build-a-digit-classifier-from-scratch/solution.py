import math

def train(X_train, y_train, X_val, y_val, n_classes):
    """
    Trains a Nearest Centroid Classifier.
    """
    # 1. Initialize structures to store the sum of pixels and count for each class
    # Each image has 64 features (8x8)
    n_features = 64
    centroids = [[0.0] * n_features for _ in range(n_classes)]
    counts = [0] * n_classes

    # 2. Accumulate pixel values for each class
    for i in range(len(X_train)):
        label = y_train[i]
        counts[label] += 1
        for j in range(n_features):
            centroids[label][j] += X_train[i][j]

    # 3. Calculate the average (mean) for each class to get the centroids
    for i in range(n_classes):
        if counts[i] > 0:
            for j in range(n_features):
                centroids[i][j] /= counts[i]

    def predict(X):
        predictions = []
        for image in X:
            best_class = -1
            min_dist = float('inf')
            
            # 4. Find the centroid with the smallest Euclidean distance
            for class_idx in range(n_classes):
                dist = 0
                for j in range(n_features):
                    diff = image[j] - centroids[class_idx][j]
                    dist += diff * diff
                
                if dist < min_dist:
                    min_dist = dist
                    best_class = class_idx
            
            predictions.append(best_class)
        return predictions

    return predict