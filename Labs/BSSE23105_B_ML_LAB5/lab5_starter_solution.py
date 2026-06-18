"""
Lab 6 Starter Code
Decision Tree Split using Gini and Information Gain
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# ============================================================
# DATASET
# ============================================================

def load_binary_iris():
    """
    Load iris dataset but keep only two classes.

    This simplifies the task to binary classification
    so that Gini impurity is easier to compute.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    data = load_iris()

    X = data.data
    y = data.target

    mask = y < 2
    X = X[mask]
    y = y[mask]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )


# ============================================================
# STUDENT TASKS
# ============================================================

def gini(y):
    """
    TODO

    Compute Gini impurity.

    Parameters
    ----------
    y : array-like

    Returns
    -------
    float
    """

    total_samples = len(y)
 
    if total_samples == 0:
        return 0.0
 
    class_counts = np.bincount(y)
 
    # Calculating the probability of each class
    class_prob=class_counts/total_samples
    # formula
    gini_impurity=1-np.sum(class_prob**2)
 
    return float(gini_impurity)



def information_gain(parent, left, right):
    """
    TODO

    Compute Information Gain.

    Parameters
    ----------
    parent : labels before split
    left : labels in left child
    right : labels in right child

    Returns
    -------
    float
    """

    total_samples=len(parent)
    no_left=len(left)
    no_right=len(right)
 
    left_weight=no_left/total_samples
    right_weight=no_right/total_samples
 
    weighted_child_impurity=(left_weight*gini(left) + right_weight*gini(right))
 
    # Gain is basically how much purer the children are as copared to the parent
    gain = gini(parent) - weighted_child_impurity
    return gain



def find_best_split(X, y):
    """
    TODO

    Implement decision tree split search.

    Steps
    -----
    1. Loop over features
    2. Generate candidate thresholds
    3. Split dataset
    4. Compute information gain
    5. Track best split

    Returns
    -------
    best_feature
    best_threshold
    best_gain
    """

    num_features=X.shape[1]
    best_feature=None
    best_threshold=None
    best_gain=-1.0
 
    for feature_index in range(num_features):
        feature_values = X[:, feature_index]
        thresholds = candidate_thresholds(feature_values)

        for threshold in thresholds:
            left_X, right_X, left_Y, right_Y = split_dataset(X, y, feature_index, threshold)
 
            if len(right_Y) == 0 or len(left_Y) == 0:
                continue
 
            gain = information_gain(y, left_Y, right_Y)
            if gain > best_gain:
    
                best_threshold=threshold
                best_gain=gain
                best_feature=feature_index

    return best_feature, best_threshold, best_gain





# ============================================================
# HELPER FUNCTIONS
# ============================================================

def split_dataset(X, y, feature, threshold):
    """
    Split dataset into left and right partitions.

    All samples with feature value <= threshold
    go to the left node.

    Returns
    -------
    X_left, X_right, y_left, y_right
    """

    mask = X[:, feature] <= threshold

    return X[mask], X[~mask], y[mask], y[~mask]


def candidate_thresholds(values):
    """
    Compute candidate thresholds for a feature.

    Thresholds are chosen as midpoints between
    consecutive sorted unique values.

    This mirrors the approach used in most
    decision tree implementations.
    """

    values = np.sort(np.unique(values))

    return (values[:-1] + values[1:]) / 2


def add_high_cardinality_feature(X):
    """
    Add deterministic high-cardinality feature.

    Each sample receives a unique value derived
    from its index.

    This allows us to observe the bias of
    decision trees toward features with many
    unique values.
    """

    N = len(X)

    index_feature = (np.arange(N) / N).reshape(-1,1)

    return np.hstack([X, index_feature])


# ============================================================
# VISUALIZATION (optional)
# ============================================================

def plot_split(X, y, feature, threshold, ax=None, title="Decision Tree Split Visualization"):
    """
    Plot the dataset (feature 0 vs feature 2) and the split boundary.

    If ax is provided, draw on that axis (enables side-by-side subplots).
    """

    import matplotlib.pyplot as plt

    made_ax = False
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
        made_ax = True

    ax.scatter(X[y == 0, 0], X[y == 0, 2], label="Class 0", marker="o")
    ax.scatter(X[y == 1, 0], X[y == 1, 2], label="Class 1", marker="x")

    if feature == 0:
        ax.axvline(x=threshold, linestyle="--", label="Split boundary")
    elif feature == 2:
        ax.axhline(y=threshold, linestyle="--", label="Split boundary")
    else:
        ax.text(
            0.5,
            0.02,
            f"Split is on feature {feature} (not shown)",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Sepal Length (feature 0)")
    ax.set_ylabel("Petal Length (feature 2)")
    ax.set_title(title)
    ax.legend()

    if made_ax:
        plt.show()


# ============================================================
# EXPERIMENT
# ============================================================

def run_split_experiment(X, y):

    feature, threshold, gain = find_best_split(X, y)

    print("Best Feature:", feature)
    print("Threshold:", threshold)
    print("Information Gain:", gain)


# ============================================================
# MAIN
# ============================================================

def main():

    train_X, test_X, train_y, test_y = load_binary_iris()

    train_X_with_index = add_high_cardinality_feature(train_X)
    index_col = train_X_with_index.shape[1] - 1

    orig_feat, orig_thr, orig_ig = find_best_split(train_X, train_y)
    aug_feat, aug_thr, aug_ig = find_best_split(train_X_with_index, train_y)

    print("Original Dataset")
    print("Best Feature:", orig_feat)
    print("Threshold:", orig_thr)
    print("Information Gain:", orig_ig)

    print("\nWith High Cardinality Feature")
    print("Best Feature:", aug_feat)
    print("Threshold:", aug_thr)
    print("Information Gain:", aug_ig)

    print("\nResult of cardinality bias experiment:")
    print("High-cardinality (index) feature preferred?:", aug_feat == index_col)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    plot_split(train_X, train_y, orig_feat, orig_thr, ax=axes[0],title=f"Original (Feat={orig_feat}, Thr={orig_thr:.2f})")
    plot_split(train_X_with_index, train_y, aug_feat, aug_thr, ax=axes[1],title=f"With index feature (Feat={aug_feat}, Thr={aug_thr:.2f})")
    plt.tight_layout()
    plt.show()




# Part A verification
    y_check = np.array([0, 0, 0, 1])

    manual = 1 - (0.75**2 + 0.25**2)
    from_function = gini(y_check)

    print("\n\nPart A: Gini Verification using y= [0, 0, 0, 1]")
    print("Manual: ", manual)
    print("Function: ", from_function)
    print("Are they both equal?: ", manual == from_function)

    



if __name__ == "__main__":
    main()