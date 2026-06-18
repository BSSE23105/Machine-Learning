"""
Lab 4 Starter Code
Hard vs Soft Margin using C, Hinge Loss and Kernels

INSTRUCTIONS
------------
You will NOT implement SVM training.
You will use sklearn.

Your tasks are:

PART A:
Observe effect of C on clean and overlapping datasets.

PART B:
Observe noise sensitivity using large vs small C.

PART C:
Implement hinge loss and compute violations.

PART D:
Use kernels to separate nonlinear data.

DO NOT MODIFY:
- dataset functions
- evaluation helpers
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.lines import Line2D

# ============================================================
# DATASETS (DETERMINISTIC)
# ============================================================

def dataset_clean():
    """
    D1: Clean Separable Dataset

    Description:
    - 10 points in 2D
    - First 5 points are positive class (+1)
    - Last 5 points are negative class (-1)
    - Fully linearly separable without overlap

    Instructions for students:
    - Use this dataset to observe SVM behaviour when data is clean
    - Suitable for Part A (effect of C)
    - Can visualize points using scatter plot with colors
    """
    X = np.array([
        [3,3],[4,3],[3,4],[5,4],[4,5],
        [1,1],[2,1],[1,2],[2,2],[2,3]
    ])
    y = np.array([1,1,1,1,1,-1,-1,-1,-1,-1])
    return X,y


def dataset_overlap():
    """
    D2: Overlapping Dataset

    Description:
    - 12 points in 2D
    - First 5 points are positive class (+1)
    - Two points in between classes to create overlap
    - Last 5 points are negative class (-1)
    - Not fully separable; some points lie in margin

    Instructions for students:
    - Use this dataset to see how SVM handles overlap
    - Observe how decision boundary changes with different C values
    - Useful for Part A (effect of C) and Part C (hinge loss analysis)
    """
    X = np.array([
        [3,3],[4,3],[3,4],[5,4],[4,5],
        [2.5,2.5],[2.7,2.3],
        [1,1],[2,1],[1,2],[2,2],[2,3]
    ])
    y = np.array([1,1,1,1,1,1,-1,-1,-1,-1,-1,-1])
    return X,y


def dataset_noisy():
    """
    D3: Noisy Dataset

    Description:
    - Based on dataset_clean
    - Two points intentionally misclassified (y values flipped)
    - Simulates noise in labels

    Instructions for students:
    - Use this dataset to see how SVM handles label noise
    - Compare model behaviour for very large vs small C
    - Useful for Part B (noise sensitivity)
    """
    X,y = dataset_clean()
    y[0] = -1  # introduce misclassification
    y[3] = -1
    return X,y

# ============================================================
# MODEL TRAINING
# ============================================================

def train_svm(X,y,C=1,kernel='linear'):
    """
    Train a SVM using sklearn.

    Instructions for students:
    - Use sklearn.svm.SVC with the specified kernel and C.
    - Fit the model to the dataset (X, y).
    - Return the trained model.
    """
    model = SVC(kernel=kernel,C=C)
    model.fit(X,y)
    return model

# ============================================================
# HINGE LOSS
# ============================================================

def hinge_loss(X,y,w,b):
    """
    Compute total hinge loss for a linear model.

    Instructions for students:
    - For each sample (x_i, y_i), compute:
        loss_i = max(0, 1 - y_i * (w^T x_i + b))
    - Sum over all samples to get total hinge loss.
    - Return the total loss.
    - Do NOT include any sklearn functions here; use only numpy.
    """
    # calculating hinge loss for each point
    total_loss = 0
    num_samples = X.shape[0]

    for i in range(num_samples):
        margin = y[i] * (np.dot(w, X[i]) + b)
        sample_loss = max(0, 1 - margin)
        total_loss = total_loss + sample_loss

    return total_loss

# ============================================================
# EXTRACT MODEL PARAMETERS
# ============================================================

def get_w_b(model):
    """
    Extract weight vector and bias from a trained linear SVM.

    Instructions for students:
    - model.coef_ contains w
    - model.intercept_ contains b
    - Return w and b as numpy arrays/scalars
    """
    w = model.coef_[0]
    b = model.intercept_[0]
    # clamp near zero weights
    w = np.where(np.abs(w) < 1e-10, 1e-6, w)
    if abs(b) < 1e-10:
        b = 1e-6
    return w,b

# ============================================================
# PART A
# ============================================================

def plot_decision_boundary(X,y,model,title):
    # setting up the grid for contour plot
    xmin = X[:,0].min() - 1
    xmax = X[:,0].max() + 1
    ymin = X[:,1].min() - 1
    ymax = X[:,1].max() + 1
    xx,yy = np.meshgrid(np.linspace(xmin,xmax,300),
                        np.linspace(ymin,ymax,300))

    grid = np.c_[xx.ravel(),yy.ravel()]
    Z = model.decision_function(grid).reshape(xx.shape)

    plt.contourf(xx,yy,Z,levels=50,cmap='RdBu',alpha=0.3)
    plt.contour(xx,yy,Z,levels=[-1,0,1],colors=['blue','black','red'],
                linestyles=['--','-','--'])

    # plot data and support vectors
    plt.scatter(X[:,0],X[:,1],c=y,cmap='bwr',edgecolors='k',s=60)
    svecs = model.support_vectors_
    plt.scatter(svecs[:,0],svecs[:,1],s=150,
                facecolors='none',edgecolors='green',linewidths=2)

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

def part_A_effect_of_C():
    """
    Investigate effect of C on clean and overlapping datasets.

    Instructions for students:
    - Load dataset_clean() and dataset_overlap()
    - Train SVM with C = 0.01, 1, 1000 (use train_svm)
    - For each model:
        - Plot decision boundary and margins
        - Compare how boundaries change with C
    - Observe how large C leads to hard-margin-like behaviour
    """
    # checking effect of different C on both datasets
    Cvals = [0.1, 1, 1.5, 2]

    data_sets = [('Clean Dataset (D1)',dataset_clean()),
                 ('Overlapping Dataset (D2)',dataset_overlap())]

    for dname,(X,y) in data_sets:
        fig,axes = plt.subplots(1,len(Cvals),figsize=(5*len(Cvals),4))
        
        fig.suptitle(f'Effect of C on {dname}',fontsize=14)

        for j,cval in enumerate(Cvals):
            plt.sca(axes[j])
            m = train_svm(X,y,C=cval,kernel='linear')
            plot_decision_boundary(X,y,m,f'C = {cval}')
        legend_handle = Line2D([0],[0],marker='o',color='w',markeredgecolor='green',markerfacecolor='none',markersize=10,markeredgewidth=2,label='Support Vectors')
        
        fig.legend(handles=[legend_handle],loc='upper right',fontsize=10)
        
        plt.tight_layout()
        plt.show()

# ============================================================
# PART B
# ============================================================

def part_B_noise_sensitivity():
    """
    Analyze noise sensitivity using D3.

    Instructions for students:
    - Load dataset_noisy()
    - Train SVM with:
        - Very large C (e.g., 1000)
        - Small C (e.g., 0.01)
    - Plot both decision boundaries on the same figure
    - Observe which one is more robust to misclassified points
    """
    # comparing small and large C on noisy data
    Xnoisy,ynoisy = dataset_noisy()
    
    c_options = [0.01, 1000]
    fig,axes = plt.subplots(1,len(c_options),figsize=(5*len(c_options),4))
    
    fig.suptitle('Noise Sensitivity on Noisy Dataset (D3)',fontsize=14)

    for j,cval in enumerate(c_options):
        plt.sca(axes[j])
        model_b = train_svm(Xnoisy,ynoisy,C=cval,kernel='linear')
        
        plot_decision_boundary(Xnoisy,ynoisy,model_b,f'C = {cval}')

    legend_handle = Line2D([0],[0],marker='o',color='w',markeredgecolor='green',markerfacecolor='none',markersize=10,markeredgewidth=2,
                           label='Support Vectors')
    fig.legend(handles=[legend_handle],loc='upper right',fontsize=10)
    
    plt.tight_layout()
    plt.show()

# ============================================================
# PART C
# ============================================================

def part_C_hinge_analysis():
    """
    Compute hinge loss for dataset_overlap.

    Instructions for students:
    - Load dataset_overlap()
    - Train SVM with several C values (e.g., 0.01, 1, 10, 100)
    - Extract w, b for each model using get_w_b()
    - Compute total hinge loss using hinge_loss()
    - Compare magnitude of hinge loss for different C
    - Optional: count number of support vectors (model.support_vectors_)
    - Optional: plot decision boundaries for visualization
    """
    # table showing hinge loss for different regularization
    Xoverlap,yoverlap = dataset_overlap()
    c_range = [0.01, 0.1, 1, 10, 100]

    print("\nHinge Loss Analysis on Overlapping Dataset")
    
    print(f"{'C':<10}{'Hinge Loss':<20}{'Support Vectors'}")

    for cval in c_range:
        model_c = train_svm(Xoverlap,yoverlap,C=cval,kernel='linear')
        weights,b_val = get_w_b(model_c)
        loss_val = hinge_loss(Xoverlap,yoverlap,weights,b_val)
        n_sv = len(model_c.support_vectors_)
        print(f"{cval:<10}{loss_val:<20.4f}{n_sv}")

    # plotting boundaries for each C
    fig,axes = plt.subplots(1,len(c_range),figsize=(5*len(c_range),4))
    fig.suptitle('Hinge Loss Analysis on Overlapping Dataset (D2)',fontsize=14)

    for j,cval in enumerate(c_range):
        model_c = train_svm(Xoverlap,yoverlap,C=cval,kernel='linear')
        weights,b_val = get_w_b(model_c)
        loss_val = hinge_loss(Xoverlap,yoverlap,weights,b_val)

        plt.sca(axes[j])
        plot_decision_boundary(Xoverlap,yoverlap,model_c,
                               f'C = {cval}\nHinge = {loss_val:.4f}')

    legend_handle = Line2D([0],[0],marker='o',color='w',markeredgecolor='green',
                           markerfacecolor='none',markersize=10,markeredgewidth=2,
                           label='Support Vectors')
    fig.legend(handles=[legend_handle],loc='upper right',fontsize=10)
    plt.tight_layout()
    plt.show()

# ============================================================
# PART D
# ============================================================

def generate_circles():
    from sklearn.datasets import make_circles
    X,y = make_circles(n_samples=200,noise=0.05,factor=0.5,random_state=0)
    y[y==0] = -1
    return X,y

def part_D_kernels():
    """
    Use kernel SVMs to separate nonlinear data.

    Instructions for students:
    - Generate concentric circles dataset using generate_circles()
    - Train SVM using:
        - linear kernel
        - polynomial kernel
        - rbf kernel
    - Plot decision boundaries for each kernel
    - Compare performance and separability
    """
    # trying all three kernels on circle data
    Xcircles,ycircles = generate_circles()
    kernels = ['linear','poly','rbf']

    fig,axes = plt.subplots(1,len(kernels),figsize=(5*len(kernels),4))
    fig.suptitle('Kernel Comparison on Concentric Circles',fontsize=14)

    for j,ktype in enumerate(kernels):
        plt.sca(axes[j])
        model_d = train_svm(Xcircles,ycircles,C=1,kernel=ktype)
        
        plot_decision_boundary(Xcircles,ycircles,model_d,f'Kernel = {ktype}')
    legend_handle = Line2D([0],[0],marker='o',color='w',markeredgecolor='green',markerfacecolor='none',markersize=10,markeredgewidth=2,
label='Support Vectors')
    fig.legend(handles=[legend_handle],loc='upper right',fontsize=10)
    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    part_A_effect_of_C()
    part_B_noise_sensitivity()
    part_C_hinge_analysis()
    part_D_kernels()