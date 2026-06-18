
import numpy as np
import matplotlib.pyplot as plt

# ================================
# DATA GENERATION
# ================================

def generate_dataset(n=10, noise_std=0.1, seed=42):
    np.random.seed(seed)
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, noise_std, size=n)
    return x.reshape(-1, 1), y.reshape(-1, 1)

# ================================
# FEATURE ENGINEERING
# ================================

def build_polynomial_features(x, degree):

    # Converting the input into a column shape with one colomn and it's rows equal to the number of samples.
    x = np.array(x).reshape(-1, 1)

    # Starting with a column of 1s (this helps the model learn the intercept)
    features = np.ones((x.shape[0], 1))

    # Adding columns with the help of loop and hstack till x, x², x³ ... up to max_degree
    for i in range(1, degree + 1):
        features = np.hstack((features, x ** i))

    return features



def plot_results(x_train, y_train, x_plot, pred_curves, labels, title="Model Fit"):
    # Task A2 --> Plot training data and fitted curve (using test points).
    # Task B2 --> Plot fitted curves for each ridge lambda value.
    plt.figure(figsize=(9, 5))
    plt.scatter(x_train, y_train, label="Training data", zorder=3)

    for yCurve, label in zip(pred_curves, labels):
        yCurve = np.asarray(yCurve).reshape(-1, 1)
        plt.plot(x_plot.flatten(), yCurve.flatten(), label=label, linewidth=2, zorder=2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# ================================
# NORMAL EQUATION
# ================================

def normal_equation(X, y):
    # Task A2 --> Implement Normal Equation solution: (X^T X) w = X^T y.
    xTx = X.T @ X
    xTy = X.T @ y
    weights = np.linalg.solve(xTx, xTy)

    return weights.reshape(-1, 1)


# ================================
# RIDGE REGRESSION
# ================================

def ridge_regression(X, y, lambda_):
    numberOfFeatures = X.shape[1]

    xTx = X.T @ X
    xTy = X.T @ y   

     # Task B1 --> Ridge closed-form solve, but do NOT regularize the bias term.
    regMatrix = float(lambda_) * np.eye(numberOfFeatures, dtype=float)
    regMatrix[0, 0] = 0.0       # bias is column 0

    weights = np.linalg.solve(xTx + regMatrix, xTy)
    return weights.reshape(-1, 1)

# ================================
# ANALYSIS UTILITIES
# ================================

def compute_condition_number(X):
    # Task A3 --> Compute condition number of X^T X (numerical stability).  
    return np.linalg.cond(X.T @ X)



def show_polynomial_example():

    xTrain, yTrain = generate_dataset(n=10, noise_std=0.1, seed=42)
    
    degree = 20

    # Task A1 --> Build polynomial feature matrix (bias + up to degree 20)
    XTrainPoly = build_polynomial_features(xTrain, degree)

    # Task A1 --> Verify dimensions
    print("\nTask A1 --> Dimension check")
    print(f"xTrain: {xTrain.shape} | yTrain: {yTrain.shape} | XTrainPoly: {XTrainPoly.shape}") #XTrainPoly = 10 * 21 (10 rows, 21 columns: bias + 20 degrees)

    # Task A2 --> Fit 20-degree polynomial using the normal equation
    wNE = normal_equation(XTrainPoly, yTrain)

    # Task A2 --> Plot fitted curve using exactly 100 test points
    xPlot = np.linspace(0.0, 1.0, 100).reshape(-1, 1)
    XPlotPoly = build_polynomial_features(xPlot, degree)
    yPredNE = XPlotPoly @ wNE

    # Task A2 --> Plot ONLY the normal-equation fit 
    plot_results(
        x_train=xTrain,
        y_train=yTrain,
        x_plot=xPlot,
        pred_curves=[yPredNE],
        labels=[f"degree {degree} (normal eq)"],
        title="Task A2: Degree-20 polynomial fit (Normal Equation)"
    )

    # Task A3 --> Print coefficient magnitudes + condition number (unregularized)
    print("\nTask A3 --> Unregularized stats")
    condNumber = compute_condition_number(XTrainPoly)
    weightSize = np.linalg.norm(wNE)
    largestWeight = np.max(np.abs(wNE))
    first10Weights = wNE.flatten()[:10]

    print(f"Condition number (X^T X): {condNumber:.4e}")
    print(f"Weight size (L2 norm)   : {weightSize:.6g}")
    print(f"Largest weight    : {largestWeight:.6g}")
    print("First 10 weights         :", first10Weights)



    lamList = [0.0, 1e-4, 1e-2, 1e-1, 1.0]
    curvesToPlot = [yPredNE]
    curveLabels = [f"degree {degree} (no reg)"]

    # Task B2 --> Ridge sweep (record coefficient norm and condition number for each lambda)
    print("\nTask B2 --> Ridge results")
    print(f"\n{'lambda':>10} | {'weight norm':>11} | {'condition number':>16}")

    XTX = XTrainPoly.T @ XTrainPoly
    nFeat = XTrainPoly.shape[1]

    for lam in lamList:
        wRidge = ridge_regression(XTrainPoly, yTrain, lam)

        curvesToPlot.append(XPlotPoly @ wRidge)
        curveLabels.append(f"ridge λ={lam:g}")

        ridgeReg = lam * np.eye(nFeat)
        ridgeReg[0, 0] = 0.0
        condReg = np.linalg.cond(XTX + ridgeReg)

        print(f"\n{lam:10g} | {np.linalg.norm(wRidge):11.4f} | {condReg:16.4e}")

    plot_results(
        x_train=xTrain,
        y_train=yTrain,
        x_plot=xPlot,
        pred_curves=curvesToPlot,
        labels=curveLabels,
        title=f"Degree {degree} polynomial: unregularized vs ridge"
    )


# ================================
# UNIT TESTS
# ================================

def test_polynomial_shape():
    x, _ = generate_dataset()
    X = build_polynomial_features(x, 20)
    assert X.shape[1] == 21, "Feature matrix should include bias + 20 degrees"

def test_normal_equation_solution():
    x, y = generate_dataset()
    X = build_polynomial_features(x, 2)
    w = normal_equation(X, y)
    assert w.shape[0] == X.shape[1], "Weight dimension mismatch"

def test_ridge_reduces_norm():
    x, y = generate_dataset()
    X = build_polynomial_features(x, 20)
    w_no_reg = normal_equation(X, y)
    w_reg = ridge_regression(X, y, 1.0)
    assert np.linalg.norm(w_reg) < np.linalg.norm(w_no_reg),         "Ridge should reduce coefficient magnitude"

def run_all_tests():
    test_polynomial_shape()
    test_normal_equation_solution()
    test_ridge_reduces_norm()
    print("All tests passed.")

if __name__ == "__main__":
    run_all_tests()
    show_polynomial_example()