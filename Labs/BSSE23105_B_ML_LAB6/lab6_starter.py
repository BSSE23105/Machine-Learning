"""
Lab 6 Starter Code
K-Means From Scratch

This code provides a skeleton for implementing K-Means clustering.
Students are expected to complete the TODO sections.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs

# ============================================================
# DATASET LOADERS
# ============================================================

def load_iris_dataset():
    """
    Load the Iris dataset.

    Tasks:
    - Select only two features: Sepal Length and Petal Length
      as specified in the lab (for 2D visualization).
    - Return the data as a NumPy array X of shape (150, 2).
    """
    data = load_iris()

    # TODO: select two features
    X = data.data[:, [0, 2]]
    return X



def generate_blobs(separable=True):
    """
    Generate synthetic clusters using sklearn.make_blobs.

    Parameters:
    - separable: if True, generate well-separated clusters
                 if False, generate overlapping clusters

    Tasks:
    - Create a dataset X with multiple clusters.
    - This allows observing K-Means behavior on different
      cluster distributions.
    """
    if separable:
        # TODO: generate well-separated blobs
       X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)
    else:
        # TODO: generate overlapping blobs
        X, _ = make_blobs(n_samples=300, centers=3, cluster_std=2.8, random_state=0)

    return X


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def centroid_shift(old_centroids, new_centroids):
    """
    Compute how much centroids moved after an update.

    Parameters:
    - old_centroids: previous centroids
    - new_centroids: updated centroids

    Tasks:
    - Compute L2 norm of the difference between old and new centroids.
    - Used as a convergence criterion in K-Means.
    """
    # TODO implement actual computation
    # Example: np.linalg.norm(old_centroids - new_centroids)
    return np.linalg.norm(old_centroids - new_centroids)


# ============================================================
# KMEANS CLASS
# ============================================================

class MyKMeans:
    """
    K-Means clustering class.

    Tasks:
    - Implement K-Means algorithm from scratch.
    - Experiment with different distance metrics: Euclidean, Manhattan, Cosine.
    - Implement cluster assignment, centroid update, and convergence check.
    """

    def __init__(self, k=3, metric="euclidean", max_iter=100):
        """
        Initialize K-Means parameters.

        Parameters:
        - k: number of clusters
        - metric: distance metric to use
        - max_iter: maximum iterations before stopping
        """
        self.k = k
        self.metric = metric
        self.max_iter = max_iter
        self.centroids = None
        self.labels_ = None
        self.random_state = 42
        self.tol = 1e-4

    # --------------------------------------------------------
    # DISTANCE FUNCTION
    # --------------------------------------------------------

    def _distance(self, x1, x2):
        """
        Compute distance between two points.

        Tasks:
        - Implement three distance metrics:
        * Euclidean: circular clusters
        * Manhattan: diamond-shaped clusters
        * Cosine: measures angle between vectors
        - Hint for cosine: distance = 1 - (dot(x1,x2)/(||x1||*||x2||))
        """
        if self.metric == "euclidean":
            diff = x1 - x2
            return np.sqrt(np.sum(diff * diff))

        elif self.metric == "manhattan":
            diff = x1 - x2
            return np.sum(np.abs(diff))

        elif self.metric == "cosine":
            dot_val = np.dot(x1, x2)

            norm1 = np.linalg.norm(x1)
            norm2 = np.linalg.norm(x2)

            if norm1 == 0 or norm2 == 0:
                return 1.0

            cos_sim = dot_val / (norm1 * norm2)
            return 1 - cos_sim

        else:
            return 0

    # --------------------------------------------------------

    def initialize_centroids(self, X):
        """
        Randomly select K points from dataset as initial centroids.

        Tasks:
        - Pick k random indices from X
        - Assign these points as initial centroids
        """
        np.random.seed(self.random_state)

        chosen_points = []

        random_indices = np.random.choice(len(X), self.k, replace=False)

        for i in random_indices:
            chosen_points.append(X[i])

        return np.array(chosen_points)

    # --------------------------------------------------------

    def assign_clusters(self, X, centroids):
        """
        Assign each point to the nearest centroid.

        Tasks:
        - Compute distance from each point to every centroid
        - Assign label based on nearest centroid
        - Return array of cluster labels
        """
        # TODO
        labels = []

        for point in X:
            distances = []

            for centroid in centroids:
                distances.append(self._distance(point, centroid))

            labels.append(np.argmin(distances))

        return np.array(labels)
    

    # --------------------------------------------------------

    def update_centroids(self, X, labels):
        """
        Update centroid positions based on cluster assignments.

        Tasks:
        - For each cluster, compute mean of points assigned to it
        - If a cluster has no points (empty), randomly reinitialize
          its centroid (safeguard)
        """
        centroids = []

        for i in range(self.k):
            # TODO: get points belonging to cluster i
            cluster_points = X[labels == i]

            if len(cluster_points) == 0:
                # SAFEGUARD: choose random point as centroid
                random_index = np.random.randint(0, len(X))
                centroid = X[random_index]
            else:
                # TODO: compute mean of cluster points
                centroid = np.mean(cluster_points, axis=0)

            centroids.append(centroid)

        return np.array(centroids)

    # --------------------------------------------------------

    def fit(self, X):
        """
        Fit the K-Means model to the data X.

        Algorithm:
        1. Initialize centroids
        2. Assign points to nearest centroid
        3. Update centroids
        4. Repeat until convergence (centroid movement < tol)
        or max iterations reached

        Tasks:
        - Implement K-Means training loop
        - Use centroid_shift to detect convergence
        """
        np.random.seed(self.random_state)
        centroids = self.initialize_centroids(X)

        for iteration in range(self.max_iter):
            labels = self.assign_clusters(X, centroids)
            new_centroids = self.update_centroids(X, labels)

            # TODO compute centroid movement
            shift = centroid_shift(centroids, new_centroids)

            centroids = new_centroids

            if shift < self.tol:
                break

        self.centroids = centroids
        self.labels_ = self.assign_clusters(X, self.centroids)

    # --------------------------------------------------------

    def predict(self, X):
        """
        Predict cluster labels for new data points.

        Tasks:
        - Compute distance from each point to centroids
        - Assign each point to nearest centroid
        """
        # TODO
        labels = []

        for point in X:
            distances = []

            for centroid in self.centroids:
                distances.append(self._distance(point, centroid))

            labels.append(np.argmin(distances))

        return np.array(labels)

    # --------------------------------------------------------
    # INERTIA
    # --------------------------------------------------------

    def compute_inertia(self, X, labels, centroids):
        """
        Compute Within-Cluster Sum of Squares (WCSS / Inertia).

        Tasks:
        - For each point, compute squared distance to its centroid
        - Sum all squared distances
        - Lower values indicate tighter clusters
        """
        # TODO
        inertia = 0

        for i, point in enumerate(X):
            center = centroids[labels[i]]
            diff = point-center
            inertia += np.dot(diff, diff)

        return inertia

# ============================================================
# VISUALIZATION
# ============================================================

def plot_clusters(X, labels, centroids, title):
    """
    Scatter plot of clusters.

    Tasks:
    - Color points based on cluster labels
    - Plot centroids as 'X' markers
    """
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=labels)

    if centroids is not None:
        plt.scatter(
            centroids[:,0],
            centroids[:,1],
            marker='X',
            s=200
        )

    plt.title(title)


def plot_decision_boundaries(model, X, title=None):
    """
    Visualize clustering regions (decision boundaries).

    Tasks:
    - Create a mesh grid over the feature space
    - Predict cluster label for each grid point
    - Plot colored decision regions
    - Useful for comparing distance metrics
    """
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),np.arange(y_min, y_max, 0.05))

    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid)

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='viridis', edgecolor='k')

    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], marker='X', s=200, c='red')

    if title:
        plt.title(title)
    else:
        plt.title(f"Decision Boundaries ({model.metric})")
    plt.show()

# ============================================================
# ELBOW METHOD
# ============================================================

def plot_elbow_curve(X, k_values):
    """
    Elbow method to determine optimal K.

    Tasks:
    - Train K-Means for multiple K values
    - Compute inertia for each K
    - Plot inertia vs K
    - Observe the 'elbow' to estimate optimal clusters
    """
    inertias = []

    for k in k_values:

        model = MyKMeans(k=k)

        model.fit(X)

        labels = model.predict(X)


        inertia = model.compute_inertia(X, labels, model.centroids)
        inertias.append(inertia)

    plt.figure()
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.grid()
    plt.show()

# ============================================================
# MAIN DEMO
# ============================================================

def main():
    """
    Main demonstration.

    Tasks:
    - Load Iris dataset
    - Run K-Means with different metrics: Euclidean, Manhattan, Cosine
    - Visualize resulting clusters
    - Run Elbow method to determine optimal K
    """
    # iris data
    X = load_iris_dataset()
    metrics = ["euclidean", "manhattan", "cosine"]

    for m in metrics:
        km = MyKMeans(k=3, metric=m)

        km.fit(X)

        plot_decision_boundaries(km, X, title=f"Iris Dataset ({m.capitalize()})")

    # elbow
    plot_elbow_curve(X, range(1, 10))

    # separable blobs
    data1 = generate_blobs(separable=True)

    km2 = MyKMeans(k=3, metric="euclidean")

    km2.fit(data1)

    plot_decision_boundaries(km2, data1, title="Separable Blobs (Euclidean)")

    # overlapping
    data2 = generate_blobs(separable=False)

    km3 = MyKMeans(k=3, metric="euclidean")

    km3.fit(data2)
    
    plot_decision_boundaries(km3, data2, title="Overlapping Blobs (Euclidean)")


if __name__ == "__main__":
    main()