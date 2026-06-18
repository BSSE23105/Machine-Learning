import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# ==========================================================
# PART A — DATASET
# ==========================================================

def generate_dataset(n=100, seed=42):
    """
    Generate a 2D linearly separable dataset.

    Requirements:
    - Use fixed random seed
    - Labels must be {-1, +1}
    - Return X (n,2), y (n,)
    """

    np.random.seed(seed)

    # Splitting in half positive and half negative
    n_positive = n // 2

    n_negative = n - n_positive

    # Classes centers
    center_positive = np.array([2.5, 2.5])

    center_negative = np.array([-2.5, -2.5])

    # Standard deviation
    spread = 0.6

    positive_points = center_positive+spread*np.random.randn(n_positive, 2)

    negative_points = center_negative+spread*np.random.randn(n_negative, 2)

    X = np.vstack([positive_points, negative_points])

    y = np.hstack([np.ones(n_positive, dtype=int), -1*np.ones(n_negative, dtype=int)])

    return X, y


# ==========================================================
# PART B — TRAIN MAXIMUM MARGIN CLASSIFIER
# ==========================================================

def train_linear_svm(X, y):
    """
    Train a linear SVM using sklearn.

    Instructions:
    - Use kernel='linear'
    - Fit the model
    - Return trained model

    DO NOT modify dataset.
    """

    # Using a very large C for identifying hard-margin SVM for separable data
    model_svm = SVC(kernel='linear', C=1e6)

    model_svm.fit(X, y)

    return model_svm 




# ==========================================================
# PART C — EXTRACT PARAMETERS
# ==========================================================

def extract_parameters(model):
    """
    Extract:
        w (weight vector)
        b (bias term)

    Hint:
        model.coef_
        model.intercept_

    Return:
        w, b
    """

    weight = model.coef_[0].copy()

    bias_term = float(model.intercept_[0])

    norm_w = np.linalg.norm(weight)

    print("  ||w||---> ", norm_w)

    return weight, bias_term


# ==========================================================
# PART D — CONSTRAINT VERIFICATION
# ==========================================================

def compute_margins(X, y, w, b):
    """
    Compute:
        y_i (w^T x_i + b)

    Return:
        array of margins
    """

    linear_scores=X.dot(w)+b

    margins=y*linear_scores

    return margins


def verify_constraints(margins):
    """
    Print:
        - Minimum margin value
        - Whether all margins >= 1

    DO NOT modify margins.
    """

    min_value = margins.min()

    # Normal strict check
    strict_check = np.all(margins >= 1.0)

    # Small tolerance is needed because floating point calculations
    # sometimes give values like 0.99999999 instead of exactly 1
    tol_check = np.all(margins >= 1.0 - 1e-8)

    print("Minimum margin value:", min_value)

    print("Without tolerance ->", strict_check)

    print("With tolerance ->", tol_check)

    print("\n")


# ==========================================================
# PART E — GEOMETRIC MARGIN WIDTH
# ==========================================================

def compute_margin_width(w):
    """
    Compute geometric margin width:

        2 / ||w||

    Return scalar.
    """

    norm_w = np.linalg.norm(w)

    if norm_w == 0:
        return float('inf')
    return 2.0 / norm_w


# ==========================================================
# PART F — SUPPORT VECTORS
# ==========================================================

def analyze_support_vectors(model, X, y, w, b):
    """
    Tasks:

    1. Print indices of support vectors.
    2. Print number of support vectors.
    3. Verify numerically that support vectors
       approximately satisfy:

            y_i (w^T x_i + b) ≈ 1

    """

    sv_indices = model.support_.copy()
    sv_points = model.support_vectors_.copy()
    num_sv = sv_points.shape[0]

    print("Support vector indices:", sv_indices)
    print("\n")

    print("Number of support vectors:", num_sv)
    print("\n")

    print("Support vector coordinates:\n", sv_points)
    print("\n")
    margins_for_sv = []

    for idx in sv_indices:
        x_i = X[idx]
        y_i = y[idx]
        margin_value = y_i*(w.dot(x_i)+b)
        margins_for_sv.append(margin_value)

    margins_for_sv = np.array(margins_for_sv)
    print("Margins for support vectors (should be ≈ 1):", margins_for_sv)
    print("All support margins approx equal to 1 (tol=1e-6):",np.allclose(margins_for_sv, np.ones_like(margins_for_sv), atol=1e-6))


# ==========================================================
# PART G — REMOVE ONE SUPPORT VECTOR
# ==========================================================

def remove_one_support_vector(model, X, y):
    """
    Remove the first support vector from dataset.

    Steps:
    1. Identify index using model.support_
    2. Remove from X and y
    3. Return new X, y

    DO NOT retrain inside this function.
    """
    if model.support_.size == 0:
        return X.copy(), y.copy()

    idx_to_remove = int(model.support_[0])

    keep = np.ones(X.shape[0], dtype=bool)

    keep[idx_to_remove] = False

    X_reduced = X[keep].copy()
    y_reduced = y[keep].copy()

    print(f"Removed support vector at original index: {idx_to_remove}")

    return X_reduced, y_reduced


# ==========================================================
# PART H — SCALING EXPERIMENT
# ==========================================================

def scale_dataset(X, factor):
    """
    Multiply features by scaling factor.

    Return scaled X.
    """

    return (X*factor).copy()


# ==========================================================
# PLOTTING UTILITY (Provided)
# ==========================================================

def plot_model(X, y, model, title="Model"):
    """
    Provided plotting function.
    Students should call this,
    not modify it.
    """

    plt.figure()

    plt.scatter(X[:, 0], X[:, 1], c=y)

    w = model.coef_[0]
    b = model.intercept_[0]

    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = -(w[0] * x_vals + b) / w[1]

    plt.plot(x_vals, y_vals)

    # Margin lines
    y_margin1 = -(w[0]*x_vals + b - 1) / w[1]
    y_margin2 = -(w[0]*x_vals + b + 1) / w[1]

    plt.plot(x_vals, y_margin1, linestyle='--')
    plt.plot(x_vals, y_margin2, linestyle='--')

    # Highlight support vectors
    plt.scatter(model.support_vectors_[:, 0],model.support_vectors_[:, 1],s=150,facecolors='none',edgecolors='black')

    plt.title(title)
    plt.show()


# ==========================================================
# MAIN BLOCK (Students Complete Sequentially)
# ==========================================================

if __name__ == "__main__":

    # Step 1
    X, y = generate_dataset()

     # Raw Visualization of data
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title("Generated Dataset")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

    # Step 2
    model = train_linear_svm(X, y)

    plot_model(X, y, model)

    # Step 3
    w, b = extract_parameters(model)

    print("\n")

    # Step 4
    margins = compute_margins(X, y, w, b)

    print("\n")
    
    verify_constraints(margins)

    print("\n")

    # Step 5
    margin_width = compute_margin_width(w)
    
    print("Geometric margin width:", margin_width)

    print("\n")

    # Step 6
    analyze_support_vectors(model, X, y, w, b)

    print("\n")

    # Step 7
    X_new, y_new = remove_one_support_vector(model, X, y)
    model_new = train_linear_svm(X_new, y_new)
    plot_model(X_new, y_new, model_new, title="After Removing Support Vector")

    print("\n")

    # Step 8
    X_scaled = scale_dataset(X, factor=10)
    model_scaled = train_linear_svm(X_scaled, y)

    w_scaled, b_scaled = extract_parameters(model_scaled)
    new_margin = compute_margin_width(w_scaled)
    print("New geometric margin width:", new_margin)

    plot_model(X_scaled, y, model_scaled, title="After Scaling")

