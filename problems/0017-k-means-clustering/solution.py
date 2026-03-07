import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    points = np.array(points, dtype=float)
    centroids = np.array(initial_centroids, dtype=float)

    for _ in range(max_iterations):
        # Compute distances from each point to each centroid
        distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)

        # Assign each point to nearest centroid
        labels = np.argmin(distances, axis=1)

        new_centroids = []
        for i in range(k):
            cluster_points = points[labels == i]

            if len(cluster_points) == 0:
                # Keep old centroid if no points assigned
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(cluster_points.mean(axis=0))

        new_centroids = np.array(new_centroids)

        # Stop early if centroids do not change
        if np.a