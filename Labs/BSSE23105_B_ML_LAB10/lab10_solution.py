"""
Lab 10 Starter Code
Model Optimization & Feature Engineering
"""

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier

# load dataset
def load_data(path="lab10_data.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


# Task 1 - Grid Search
def run_grid_search(X, y):
    """
    TODO:
    - Define param_grid
    - Use GridSearchCV
    - Return best estimator
    """
    params = {
        'max_depth': [3,5,7,10,None],
        'min_samples_split': [2,5,10,20]
    }
    clf = DecisionTreeClassifier(random_state=42)
    gs =GridSearchCV(clf, params, cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X, y)
    print("Grid Search Best Params:", gs.best_params_)
    print("Grid Search Best Score:", gs.best_score_)
    return gs.best_estimator_


# Task 2 - Random Search
def run_random_search(X, y):
    """
    TODO:
    - Define param_dist
    - Use RandomizedSearchCV
    - Return best estimator
    """
    dist = {
        'max_depth': [3,5,7,10,None],
        'min_samples_split': [2,5,10,20]
    }
    tree =DecisionTreeClassifier(random_state=42)
    rs = RandomizedSearchCV(tree, dist, n_iter=10, cv=5,
                            scoring='accuracy', random_state=42, n_jobs=-1)
    rs.fit(X, y)
    print("Random Search Best Params:", rs.best_params_)
    print("Random Search Best Score:", rs.best_score_)
    return rs.best_estimator_


# Task 3 - Bayesian Optimization with Optuna
def run_optuna(X, y):
    """
    TODO:
    - Define objective
    - Run study.optimize
    - Return best params
    """
    def obj_fn(trial):
        depth =trial.suggest_int('max_depth', 2, 15)
        min_split= trial.suggest_int('min_samples_split', 2, 20)
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_split, random_state=42)
        res =cross_val_score(dt, X, y, cv=5, scoring='accuracy')
        return res.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    stdy = optuna.create_study(direction='maximize')
    stdy.optimize(obj_fn, n_trials=20, show_progress_bar=False)
    print("Optuna Best Params:", stdy.best_params)
    print("Optuna Best Score:", stdy.best_value)
    return stdy.best_params


# Task 4 - K-Fold Cross Validation
def evaluate_cv(model, X, y):
    """
    Returns:
        mean_accuracy, std_accuracy
    """
    cv_scores =cross_val_score(model, X, y, cv=5, scoring='accuracy')
    avg_acc = float(cv_scores.mean())
    std_acc =float(cv_scores.std())
    print(f"CV Mean Accuracy: {avg_acc:.4f} (+/- {std_acc:.4f})")
    return avg_acc, std_acc


# Task 6 - Naive Target Encoding
def target_encode(df, column, target):
    """
    Naive target encoding (leakage-prone)
    """
    avg_map =df.groupby(column)[target].mean()
    return df[column].map(avg_map)


# Task 7 - K-Fold Target Encoding
def kfold_target_encode(df, column, target, n_splits=5):
    """
    Leakage-safe encoding
    """
    result = pd.Series(np.nan, index=df.index)
    splitter =KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, vl_idx in splitter.split(df):
        grp_means =df.iloc[tr_idx].groupby(column)[target].mean()
        result.iloc[vl_idx] = df.iloc[vl_idx][column].map(grp_means)
    # fill leftover nans with global mean
    result.fillna(df[target].mean(), inplace=True)
    return result


# Task 10 - Cyclical Feature Encoding
def encode_cyclical(feature, max_val):
    """
    Returns:
        sin_feature, cos_feature
    """
    s_val = np.sin(2*np.pi*feature/max_val)
    c_val =np.cos(2*np.pi*feature/max_val)
    return s_val, c_val


if __name__ == "__main__":
    import time
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    raw_df =pd.read_csv("lab10_data.csv")
    labels = raw_df["target"]
    print("Dataset shape:", raw_df.shape)
    print(raw_df.head())

    # drop leak + city for numeric tasks
    feats = raw_df.drop(columns=["target","leak_feature","city"])

    # ---- TASK 1 ----
    print("\nTASK 1: Grid Search\n")
    tick =time.time()
    gs_best =run_grid_search(feats, labels)
    tock= time.time()
    print(f"Time taken: {tock-tick:.3f}s")

    # ---- TASK 2 ----
    print("\nTASK 2: Random Search\n")
    tick =time.time()
    rs_best= run_random_search(feats, labels)
    tock =time.time()
    print(f"Time taken: {tock-tick:.3f}s")

    # ---- TASK 3 ----
    print("\nTASK 3: Bayesian Optimization (Optuna)\n")
    opt_params =run_optuna(feats, labels)

    # ---- TASK 4 ----
    print("\nTASK 4: K-Fold Cross Validation\n")
    base_tree = DecisionTreeClassifier(random_state=42)
    m_acc, s_acc= evaluate_cv(base_tree, feats, labels)

    # ---- TASK 5 ----
    print("\nTASK 5: Train/Test Split vs Cross-Validation\n")
    xtr, xte, ytr, yte =train_test_split(feats, labels, test_size=0.2, random_state=42)
    tt_tree =DecisionTreeClassifier(random_state=42)
    tt_tree.fit(xtr, ytr)
    tt_acc= accuracy_score(yte, tt_tree.predict(xte))
    print(f"Train/Test Split Accuracy: {tt_acc:.4f}")
    cv_tree = DecisionTreeClassifier(random_state=42)
    cv_m, cv_s =evaluate_cv(cv_tree, feats, labels)
    print(f"Cross-Validation Accuracy: {cv_m:.4f} (+/- {cv_s:.4f})")

    # ---- TASK 6 ----
    print("\nTASK 6: Naive Target Encoding\n")
    df1 =raw_df.copy()
    df1["city_enc"] =target_encode(df1, "city", "target")
    print("Encoded city (first 10 rows):")
    print(df1[["city","city_enc","target"]].head(10))
    naive_feats = df1[["age","income","hour","city_enc"]]
    xtr1, xte1, ytr1, yte1 =train_test_split(naive_feats, labels, test_size=0.2, random_state=42)
    nv_tree = DecisionTreeClassifier(random_state=42)
    nv_tree.fit(xtr1, ytr1)
    nv_acc =accuracy_score(yte1, nv_tree.predict(xte1))
    print(f"\nNaive Encoding Accuracy: {nv_acc:.4f}")

    # ---- TASK 7 ----
    print("\nTASK 7: K-Fold Target Encoding\n")
    df2= raw_df.copy()
    df2["city_enc"] =kfold_target_encode(df2, "city", "target", n_splits=5)
    print("K-Fold Encoded city (first 10 rows):")
    print(df2[["city","city_enc","target"]].head(10))
    kf_feats =df2[["age","income","hour","city_enc"]]
    xtr2, xte2, ytr2, yte2 =train_test_split(kf_feats, labels, test_size=0.2, random_state=42)
    kf_tree =DecisionTreeClassifier(random_state=42)
    kf_tree.fit(xtr2, ytr2)
    kf_acc= accuracy_score(yte2, kf_tree.predict(xte2))
    print(f"\nK-Fold Encoding Accuracy: {kf_acc:.4f}")
    print(f"Naive Encoding Accuracy:  {nv_acc:.4f}")
    print(f"Difference: {abs(nv_acc-kf_acc):.4f}")

    # ---- TASK 8 & 9 ----
    print("\nTASK 8 & 9: Data Leakage Detection and Fix\n")
    # with leakage (age + leak_feature only)
    leaked = raw_df[["age","leak_feature"]]
    xtr_l, xte_l, ytr_l, yte_l= train_test_split(leaked, labels, test_size=0.2, random_state=42)
    lk_tree =DecisionTreeClassifier(random_state=42)
    lk_tree.fit(xtr_l, ytr_l)
    lk_acc =accuracy_score(yte_l, lk_tree.predict(xte_l))
    print(f"WITH leak_feature -> Accuracy: {lk_acc:.4f}")

    # without leakage (age only)
    cleaned =raw_df[["age"]]
    xtr_c, xte_c, ytr_c, yte_c =train_test_split(cleaned, labels, test_size=0.2, random_state=42)
    cl_tree= DecisionTreeClassifier(random_state=42)
    cl_tree.fit(xtr_c, ytr_c)
    cl_acc =accuracy_score(yte_c, cl_tree.predict(xte_c))
    print(f"WITHOUT leak_feature -> Accuracy: {cl_acc:.4f}")
    print(f"\nCorrelation of leak_feature with target: {raw_df['leak_feature'].corr(raw_df['target']):.4f}")
    print("leak_feature is clearly derived from target (leakage!)")

    # ---- TASK 10 ----
    print("\nTASK 10: Cyclical Encoding of Hour\n")
    hrs =raw_df["hour"].values
    h_s, h_c =encode_cyclical(hrs, 24)
    print("Sample hour -> sin, cos:")
    for h in range(0,24,4):
        pos = np.where(hrs==h)[0][0]
        print(f"  hour={h:2d}  -> sin={h_s[pos]:.4f}, cos={h_c[pos]:.4f}")

    # ---- TASK 11 ----
    print("\nTASK 11: Before vs After Cyclical Encoding\n")
    # before (age + raw hour)
    raw_hr = raw_df[["age","hour"]]
    xtr_r, xte_r, ytr_r, yte_r =train_test_split(raw_hr, labels, test_size=0.2, random_state=42)
    rw_tree = DecisionTreeClassifier(random_state=42)
    rw_tree.fit(xtr_r, ytr_r)
    rw_acc =accuracy_score(yte_r, rw_tree.predict(xte_r))

    # after (age + sin/cos hour)
    cyc_df =raw_df[["age"]].copy()
    cyc_df["hr_sin"] =h_s
    cyc_df["hr_cos"]= h_c
    xtr_cy, xte_cy, ytr_cy, yte_cy =train_test_split(cyc_df, labels, test_size=0.2, random_state=42)
    cy_tree= DecisionTreeClassifier(random_state=42)
    cy_tree.fit(xtr_cy, ytr_cy)
    cy_acc =accuracy_score(yte_cy, cy_tree.predict(xte_cy))

    print(f"Before (raw hour):      Accuracy = {rw_acc:.4f}")
    print(f"After (sin/cos hour):   Accuracy = {cy_acc:.4f}")
    print(f"Improvement:            {cy_acc-rw_acc:+.4f}")