# lab9_starter.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


# ==========================================================
# DATASETS (PROVIDED - DO NOT MODIFY)
# ==========================================================

def load_dataset_60_40():
    np.random.seed(42)

    normal = np.random.multivariate_normal(
        mean=[50, 50],
        cov=[[200, 80], [80, 200]],
        size=600
    )

    fraud = np.random.multivariate_normal(
        mean=[55, 55],  # VERY close mean
        cov=[[200, 80], [80, 200]],
        size=400
    )

    X = np.vstack([normal, fraud])
    y = np.array([0]*600 + [1]*400)

    return X, y


def load_dataset_90_10():
    np.random.seed(42)

    normal = np.random.multivariate_normal(
        mean=[50, 50],
        cov=[[300, 120], [120, 300]],
        size=900
    )

    fraud = np.random.multivariate_normal(
        mean=[51, 51],
        cov=[[300, 120], [120, 300]],
        size=100
    )

    X = np.vstack([normal, fraud])
    y = np.array([0]*900 + [1]*100)

    return X, y


# ==========================================================
# CORE TASKS
# ==========================================================

# helper to draw points on a subplot
def draw_points(axis, data, target, name):
    legit = data[target == 0]
    suspicious = data[target == 1]
    axis.scatter(legit[:, 0], legit[:, 1], alpha=0.7, label="Normal", c="#1f77b4", edgecolors='black', linewidths=0.4, s=30)
    axis.scatter(suspicious[:, 0], suspicious[:, 1], alpha=0.8, label="Fraud", c="#e03131", edgecolors='black', linewidths=0.4, s=30)
    axis.set_title(name, fontweight='bold')
    axis.set_xlabel("X1")
    axis.set_ylabel("X2")
    axis.legend()
    axis.grid(True, linestyle='--', alpha=0.3)


def show_one_plot(data, target, name):
    fig, ax = plt.subplots(figsize=(7, 5))
    draw_points(ax, data, target, name)
    plt.tight_layout()
    plt.show()


def show_side_by_side(d1, t1, n1, d2, t2, n2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    draw_points(axes[0], d1, t1, n1)
    draw_points(axes[1], d2, t2, n2)
    plt.tight_layout()
    plt.show()


def train_logistic(X, y):
    """
    Train logistic regression using cross-entropy loss.
    """
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X, y)
    return logreg


def evaluate_model(model, X, y):
    """
    Return accuracy and F1-score.
    """
    predicted = model.predict(X)
    return accuracy_score(y, predicted), f1_score(y, predicted, zero_division=0)


def get_all_metrics(model, X, y):
    """get all four metrics for printing in main"""
    predicted = model.predict(X)
    acc = accuracy_score(y, predicted)
    prec = precision_score(y, predicted, zero_division=0)
    rec = recall_score(y, predicted, zero_division=0)
    f1 = f1_score(y, predicted, zero_division=0)
    return acc, prec, rec, f1


def apply_smote(X, y):
    """
    Apply SMOTE.
    """
    resampler = SMOTE(random_state=42)
    X_new, y_new = resampler.fit_resample(X, y)
    return X_new, y_new


def apply_smoteenn(X, y):
    """
    Apply SMOTEENN.
    """
    resampler = SMOTEENN(random_state=42)
    X_new, y_new = resampler.fit_resample(X, y)
    return X_new, y_new


def train_weighted_logistic(X, y):
    """
    Train logistic regression with class weights.
    """
    logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    logreg.fit(X, y)
    return logreg





# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    # task 1 - exploring the balanced-ish dataset
    print("\n--- Task 1 ---")
    data_bal, labels_bal = load_dataset_60_40()
    count_legit = int(np.sum(labels_bal == 0))
    count_fraud = int(np.sum(labels_bal == 1))
    print(f"Dataset 1 has {len(labels_bal)} samples ({count_legit} normal, {count_fraud} fraud)")

    # task 2 - now the heavily imbalanced one
    print("\n--- Task 2 ---")
    data_imb, labels_imb = load_dataset_90_10()
    count_legit_2 = int(np.sum(labels_imb == 0))
    count_fraud_2 = int(np.sum(labels_imb == 1))
    print(f"Dataset 2 has {len(labels_imb)} samples ({count_legit_2} normal, {count_fraud_2} fraud)")
    print("clearly way more imbalanced compared to dataset 1")

    # comparing both datasets visually
    show_side_by_side(data_bal, labels_bal, "Dataset 1 (60:40)",
                      data_imb, labels_imb, "Dataset 2 (90:10)")

    # task 3 - fitting logistic regression on both
    print("\n--- Task 3 & 4 ---")
    model_bal = train_logistic(data_bal, labels_bal)
    acc_bal, prec_bal, rec_bal, f1_bal = get_all_metrics(model_bal, data_bal, labels_bal)
    print(f"[Dataset 1] acc: {acc_bal:.4f} | precision: {prec_bal:.4f} | recall: {rec_bal:.4f} | f1: {f1_bal:.4f}")

    model_imb = train_logistic(data_imb, labels_imb)
    acc_imb, prec_imb, rec_imb, f1_imb = get_all_metrics(model_imb, data_imb, labels_imb)
    print(f"[Dataset 2] acc: {acc_imb:.4f} | precision: {prec_imb:.4f} | recall: {rec_imb:.4f} | f1: {f1_imb:.4f}")
    print("the model gets 90% accuracy on dataset 2 but its basically predicting everything as normal")

    # task 5 - SMOTE oversampling
    print("\n--- Task 5 ---")
    data_smote, labels_smote = apply_smote(data_imb, labels_imb)
    print(f"before smote: {count_legit_2} normal, {count_fraud_2} fraud")
    print(f"after smote:  {int(np.sum(labels_smote==0))} normal, {int(np.sum(labels_smote==1))} fraud")

    show_side_by_side(data_imb, labels_imb, "Before SMOTE",
                      data_smote, labels_smote, "After SMOTE")

    model_smote = train_logistic(data_smote, labels_smote)
    acc_smo, prec_smo, rec_smo, f1_smo = get_all_metrics(model_smote, data_smote, labels_smote)
    print(f"[SMOTE]     acc: {acc_smo:.4f} | precision: {prec_smo:.4f} | recall: {rec_smo:.4f} | f1: {f1_smo:.4f}")

    # task 6 - SMOTEENN (oversampling + cleaning noisy samples)
    print("\n--- Task 6 ---")
    data_se, labels_se = apply_smoteenn(data_imb, labels_imb)
    print(f"before smoteenn: {count_legit_2} normal, {count_fraud_2} fraud")
    print(f"after smoteenn:  {int(np.sum(labels_se==0))} normal, {int(np.sum(labels_se==1))} fraud")

    show_side_by_side(data_smote, labels_smote, "SMOTE",
                      data_se, labels_se, "SMOTEENN")

    model_se = train_logistic(data_se, labels_se)
    acc_se, prec_se, rec_se, f1_se = get_all_metrics(model_se, data_se, labels_se)
    print(f"[SMOTEENN]  acc: {acc_se:.4f} | precision: {prec_se:.4f} | recall: {rec_se:.4f} | f1: {f1_se:.4f}")
    print("smoteenn gives better f1 than plain smote because it also removes noisy overlap points")

    # task 7 - cost sensitive approach using weighted loss
    print("\n--- Task 7 ---")
    model_wt = train_weighted_logistic(data_imb, labels_imb)
    acc_wt, prec_wt, rec_wt, f1_wt = get_all_metrics(model_wt, data_imb, labels_imb)
    print(f"[Weighted]  acc: {acc_wt:.4f} | precision: {prec_wt:.4f} | recall: {rec_wt:.4f} | f1: {f1_wt:.4f}")
    print("class_weight=balanced makes the model pay more attention to the minority class")

    # quick summary table at the end
    print("\n\n--- Summary (all methods on Dataset 2) ---\n")
    rows = [
        ("Baseline LR",   acc_imb, prec_imb, rec_imb, f1_imb),
        ("With SMOTE",     acc_smo, prec_smo, rec_smo, f1_smo),
        ("With SMOTEENN",  acc_se,  prec_se,  rec_se,  f1_se),
        ("Weighted LR",    acc_wt,  prec_wt,  rec_wt,  f1_wt),
    ]
    print(f"  {'Method':<18} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    for r in rows:
        print(f"  {r[0]:<18} {r[1]:>7.3f} {r[2]:>7.3f} {r[3]:>7.3f} {r[4]:>7.3f}")

    print("\ndone.")