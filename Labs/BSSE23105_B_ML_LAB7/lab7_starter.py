"""
Lab 7 Starter Code
Agglomerative Clustering from Scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# ============================================================
# DATASET
# ============================================================

def generate_dataset():

    X, _ = make_blobs(
        n_samples=100,
        centers=3,
        cluster_std=1.2,
        random_state=42
    )
    return X


# ============================================================
# PART A
# ============================================================

def compute_distance_matrix(X):

    # TODO
    number_of_datapoints = len(X)

    distance_matrix_result = np.zeros((number_of_datapoints, number_of_datapoints))

    for row_index in range(number_of_datapoints):

        for column_index in range(number_of_datapoints):

            first_point = X[row_index]

            second_point = X[column_index]

            difference_vector =first_point- second_point

            euclidean_distance_val = np.sqrt(np.sum(difference_vector **2))

            distance_matrix_result[row_index][column_index] =euclidean_distance_val
    return distance_matrix_result


# ============================================================
# PART B
# ============================================================

def linkage_distance(cluster1, cluster2, X, method="single"):

    # TODO
    all_distances_between_clusters = []

    for point_index_cluster1 in cluster1:

        for point_index_cluster2 in cluster2:

            pointA = X[point_index_cluster1]

            pointB =X[point_index_cluster2]

            distance_between_points = np.sqrt(np.sum((pointA- pointB)**2))

            all_distances_between_clusters.append(distance_between_points)

    if method =="single":

        final_linkage_distance = np.min(all_distances_between_clusters)

    elif method=="complete":

        final_linkage_distance =np.max(all_distances_between_clusters)

    elif method == "average":

        final_linkage_distance =np.mean(all_distances_between_clusters)

    else:
        raise ValueError(f"Unknown linkage method: {method}")
    return final_linkage_distance

# ============================================================
# PART C
# ============================================================

def find_closest_clusters(clusters, X, method):

    # TODO
    minimum_distance_found = np.inf

    closest_cluster_i = None

    closest_cluster_j =None

    total_number_of_clusters = len(clusters)

    for index_i in range(total_number_of_clusters):

        for index_j in range(index_i +1, total_number_of_clusters):

            current_cluster_one = clusters[index_i]

            current_cluster_two =clusters[index_j]

            computed_distance = linkage_distance(current_cluster_one, current_cluster_two, X, method)

            if computed_distance <minimum_distance_found:

                minimum_distance_found= computed_distance

                closest_cluster_i =index_i

                closest_cluster_j = index_j

    return closest_cluster_i, closest_cluster_j, minimum_distance_found



# ============================================================
# MODEL
# ============================================================

class MyAgglomerative:

    def __init__(self, n_clusters=3, linkage="single"):

        self.n_clusters = n_clusters

        self.linkage = linkage

        self.labels_ = None

        self.history_ = []


    def fit(self, X):

        total_samples = len(X)

        # TODO initialize clusters
        all_clusters = [[i] for i in range(total_samples)]

        while len(all_clusters) > self.n_clusters:

            # TODO find closest clusters
            idx_i, idx_j, merge_distance = find_closest_clusters(all_clusters, X, self.linkage)

            # TODO merge clusters
            merged_new_cluster = all_clusters[idx_i] +all_clusters[idx_j]

            # TODO update clusters
            all_clusters = [c for cluster_idx, c in enumerate(all_clusters) if cluster_idx!=idx_i and cluster_idx !=idx_j]

            all_clusters.append(merged_new_cluster)

            # store merge
            self.history_.append((idx_i, idx_j, merge_distance))

        # TODO assign labels
        label_array = np.zeros(total_samples, dtype=int)

        for cluster_label, points_in_this_cluster in enumerate(all_clusters):

            for single_point_index in points_in_this_cluster:

                label_array[single_point_index]= cluster_label

        self.labels_ =label_array


    def predict(self, X):
        return self.labels_

# ============================================================
# VISUALIZATION
# ============================================================

def plot_clusters(X, labels, title):

    plt.scatter(X[:,0], X[:,1], c=labels)

    plt.title(title)


# ============================================================
# DENDROGRAM (BONUS)
# ============================================================

def plot_dendrogram(X):

    # TODO
    import scipy.cluster.hierarchy as sch

    linkage_computation_result = sch.linkage(X, method="single")

    plt.figure(figsize=(10, 5))

    sch.dendrogram(linkage_computation_result)

    plt.title("Dendrogram (Single Linkage)")

    plt.xlabel("Sample Index")

    plt.ylabel("Distance")

    plt.tight_layout()

    plt.savefig("dendrogram.png", dpi=150, bbox_inches='tight')

    plt.show()


def compare_with_sklearn(X):

    from sklearn.cluster import AgglomerativeClustering

    linkage_methods_to_test = ["single", "complete","average"]

    fig, axes_grid = plt.subplots(2, 3, figsize=(15, 8))

    for column_idx, current_method_name in enumerate(linkage_methods_to_test):

        # our implementation
        my_clustering_model = MyAgglomerative(n_clusters=3, linkage=current_method_name)

        my_clustering_model.fit(X)

        my_predicted_labels =my_clustering_model.predict(X)

        axes_grid[0][column_idx].scatter(X[:,0], X[:,1], c=my_predicted_labels, cmap='viridis')

        axes_grid[0][column_idx].set_title(f"Mine - {current_method_name}")

        # sklearn
        sklearn_model = AgglomerativeClustering(n_clusters=3, linkage=current_method_name)

        sklearn_predicted_labels = sklearn_model.fit_predict(X)

        axes_grid[1][column_idx].scatter(X[:,0], X[:,1], c=sklearn_predicted_labels, cmap='viridis')

        axes_grid[1][column_idx].set_title(f"sklearn - {current_method_name}")

    plt.suptitle("My Implementation vs sklearn", fontsize=14)

    plt.tight_layout()

    plt.savefig("sklearn_comparison.png", dpi=150, bbox_inches='tight')

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    X = generate_dataset()

    linkage_methods_list = ["single", "complete", "average"]

    plt.figure(figsize=(15, 4))

    for plot_number, method_name in enumerate(linkage_methods_list):

        clustering_model = MyAgglomerative(n_clusters=3, linkage=method_name)

        clustering_model.fit(X)

        predicted_labels = clustering_model.predict(X)

        plt.subplot(1, 3, plot_number+1)

        plot_clusters(X, predicted_labels, f"Agglomerative ({method_name} linkage)")

    plt.tight_layout()

    plt.savefig("linkage_comparison.png", dpi=150, bbox_inches='tight')

    plt.show()

    # bonus: dendrogram
    plot_dendrogram(X)

    # bonus: compare with sklearn
    compare_with_sklearn(X)


if __name__ == "__main__":
    main()