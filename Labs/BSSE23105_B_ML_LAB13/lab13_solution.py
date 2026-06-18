"""
Lab 13: Drift Detection in Streaming Data
==========================================
Department of Computer and Software Engineering
SE: Machine Learning

Instructions
------------
- Complete EVERY function marked with # TODO.
- Do NOT rename any function or change its parameters or return types.
- Do NOT remove or reorder the import statements.
- You may add private helper functions anywhere below the imports.
- Run   pytest test_student.py -v   to check your work before submission.

Dataset
-------
File   : transactions_with_drift.xlsx
Sheets :
    All_Transactions           full 6-month dataset  (use for batch splitting)
    Baseline_M1_M3             months 1-3 only        (use as reference)
    Drift_Summary_Ground_Truth per-month true stats   (check AFTER all tasks)

Required packages
-----------------
    pip install pandas numpy scipy scikit-learn openpyxl matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Constants — do not change these
# ------------------------------------------------------------------
DATASET_PATH = "transactions_with_drift.xlsx"
FEATURES     = ["transaction_amount", "customer_age",
                "transaction_hour",   "device_risk_score"]
TARGET       = "is_fraud"


# ==================================================================
# PART A — STREAMING DATA SIMULATION
# ==================================================================

# grabs mean, std, min, max for each feature and the overall fraud rate from months 1-3
def task1_baseline_stats(baseline_df):
    info = {}
    for feat in FEATURES:
        vals = baseline_df[feat]
        info[feat] = {
            'mean': vals.mean(),
            'std': vals.std(),
            'min': vals.min(),
            'max': vals.max()
        }
    info['fraud_rate'] = baseline_df[TARGET].mean()
    return info


# splits the full dataset by month column and plots how avg amount and avg hour change over time
def task2_split_batches(all_df):
    chunks = {}
    for m in range(1, 7):
        chunks[m] = all_df[all_df['month'] == m].copy()

    month_nums = list(range(1, 7))
    avg_amt = [chunks[m]['transaction_amount'].mean() for m in month_nums]
    avg_hr = [chunks[m]['transaction_hour'].mean() for m in month_nums]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6))

    top.plot(month_nums, avg_amt, marker='o')
    top.set_title('Mean Transaction Amount per Month')
    top.set_xlabel('Month')
    top.set_ylabel('Mean Amount')

    bottom.plot(month_nums, avg_hr, marker='o', color='orange')
    bottom.set_title('Mean Transaction Hour per Month')
    bottom.set_xlabel('Month')
    bottom.set_ylabel('Mean Hour')

    plt.tight_layout()
    plt.savefig('task2_batch_means.png')
    plt.close()

    return chunks


# ==================================================================
# PART B — THRESHOLD-BASED DRIFT DETECTION
# ==================================================================

# checks if each month's avg transaction amount drifted beyond 2 standard deviations from baseline
def task3_mean_shift_detection(batches, baseline_stats):
    ref_mu = baseline_stats['transaction_amount']['mean']
    ref_sigma = baseline_stats['transaction_amount']['std']

    alerts = {}
    for m in range(1, 7):
        curr_mean = batches[m]['transaction_amount'].mean()
        alerts[m] = bool(abs(curr_mean - ref_mu) > 2 * ref_sigma)
    return alerts


# builds a detailed log for months that triggered a drift alert showing how far they shifted
def task4_drift_log(batches, baseline_stats, alerts):
    ref_mu = baseline_stats['transaction_amount']['mean']
    ref_sigma = baseline_stats['transaction_amount']['std']

    log = []
    for m in range(1, 7):
        if not alerts[m]:
            continue
        grp = batches[m]
        grp_mean = grp['transaction_amount'].mean()
        gap = abs(grp_mean - ref_mu)
        num_sigmas = gap / ref_sigma
        label = grp['drift_phase'].iloc[0]

        log.append({
            'month': m,
            'drift_phase': label,
            'batch_mean': grp_mean,
            'shift_magnitude': gap,
            'sigmas_away': num_sigmas
        })

    # printing the results in a readable format
    print(f"{'Month':<8}{'Drift Phase':<18}{'Batch Mean':<14}{'Shift':<12}{'Sigmas':<8}")
    print('-' * 60)
    for row in log:
        print(f"{row['month']:<8}{row['drift_phase']:<18}"
              f"{row['batch_mean']:<14.2f}{row['shift_magnitude']:<12.2f}"
              f"{row['sigmas_away']:<8.2f}")

    return log


# ==================================================================
# PART C — KS TEST
# ==================================================================

# runs the kolmogorov-smirnov test comparing each month's amounts against the baseline distribution
def task5_ks_test(batches, baseline_df):
    ref_amounts = baseline_df['transaction_amount']

    ks_results = {}
    for m in range(1, 7):
        monthly_amounts = batches[m]['transaction_amount']
        statistic, pval = ks_2samp(ref_amounts, monthly_amounts)
        ks_results[m] = {
            'ks_stat': statistic,
            'p_value': pval,
            'drifted': bool(pval < 0.05)
        }
    return ks_results


# puts mean-shift alerts and ks test results next to each other so we can compare both methods
def task6_method_comparison(alerts, ks_results):
    comparison = []
    for m in range(1, 7):
        comparison.append({
            'month': m,
            'mean_shift_alert': alerts[m],
            'ks_alert': ks_results[m]['drifted'],
            'ks_stat': ks_results[m]['ks_stat'],
            'p_value': ks_results[m]['p_value']
        })

    print(f"{'Month':<8}{'Mean-Shift':<14}{'KS Alert':<12}{'KS Stat':<12}{'p-value':<12}")
    print('-' * 58)
    for row in comparison:
        print(f"{row['month']:<8}{str(row['mean_shift_alert']):<14}"
              f"{str(row['ks_alert']):<12}{row['ks_stat']:<12.4f}{row['p_value']:<12.4f}")

    return comparison


# ==================================================================
# PART D — CONCEPT DRIFT
# ==================================================================

# trains logistic regression on baseline data with 80/20 split and reports accuracy and f1
def task7_train_baseline_model(baseline_df):
    X = baseline_df[FEATURES]
    y = baseline_df[TARGET]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_tr_scaled = sc.fit_transform(X_tr)
    X_val_scaled = sc.transform(X_val)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr_scaled, y_tr)

    preds = clf.predict(X_val_scaled)
    acc = accuracy_score(y_val, preds)
    f1_val = f1_score(y_val, preds, zero_division=0)

    return clf, sc, acc, f1_val


# applies the trained model on each monthly batch without retraining to see how performance drops
def task8_evaluate_on_batches(model, scaler, batches):
    results = {}
    for m in range(1, 7):
        grp = batches[m]
        X_scaled = scaler.transform(grp[FEATURES])
        actual = grp[TARGET]
        preds = model.predict(X_scaled)

        results[m] = {
            'accuracy': accuracy_score(actual, preds),
            'f1': f1_score(actual, preds, zero_division=0)
        }
    return results


# ==================================================================
# PART E — VISUALIZATION
# ==================================================================

# creates a 4-panel dashboard showing amount means, ks stats, fraud rate and f1 with drift shading
def task9_drift_dashboard(batches, baseline_stats, alerts,
                           ks_results, performance):
    month_nums = list(range(1, 7))

    avg_amts = [batches[m]['transaction_amount'].mean() for m in month_nums]
    ks_vals = [ks_results[m]['ks_stat'] for m in month_nums]
    fraud_pcts = [batches[m]['is_fraud'].mean() for m in month_nums]
    f1_vals = [performance[m]['f1'] for m in month_nums]

    mu = baseline_stats['transaction_amount']['mean']
    sigma = baseline_stats['transaction_amount']['std']

    fig, ax = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    # shading helper to mark drift zones on each subplot
    def mark_drift(a):
        a.axvspan(3.5, 5.5, color='lightyellow', alpha=0.6, label='Feature Drift')
        a.axvspan(5.5, 6.5, color='mistyrose', alpha=0.6, label='Concept Drift')

    # subplot 1 - transaction amount means with 2-sigma alert bounds
    ax[0].plot(month_nums, avg_amts, marker='o', color='steelblue')
    ax[0].axhline(mu + 2 * sigma, linestyle='--', color='red', label='upper bound')
    ax[0].axhline(mu - 2 * sigma, linestyle='--', color='red', label='lower bound')
    ax[0].set_ylabel('Mean Amount')
    ax[0].set_title('Mean Transaction Amount per Month')
    mark_drift(ax[0])
    ax[0].legend(fontsize=8)

    # subplot 2 - ks statistic values with drifted months annotated
    ax[1].plot(month_nums, ks_vals, marker='s', color='darkorange')
    ax[1].set_ylabel('KS Statistic')
    ax[1].set_title('KS Statistic per Month')
    mark_drift(ax[1])
    for m in month_nums:
        if ks_results[m]['drifted']:
            ax[1].annotate('Drifted', (m, ks_vals[m - 1]),
                           textcoords='offset points', xytext=(0, 10),
                           fontsize=7, ha='center', color='red')

    # subplot 3 - fraud rate trend across months
    ax[2].plot(month_nums, fraud_pcts, marker='^', color='green')
    ax[2].set_ylabel('Fraud Rate')
    ax[2].set_title('Fraud Rate per Month')
    mark_drift(ax[2])

    # subplot 4 - model f1 score showing degradation under drift
    ax[3].plot(month_nums, f1_vals, marker='D', color='purple')
    ax[3].set_ylabel('F1 Score')
    ax[3].set_title('Model F1-Score per Month')
    ax[3].set_xlabel('Month')
    mark_drift(ax[3])

    plt.tight_layout()
    plt.savefig('task9_drift_dashboard.png')
    plt.close()


# ==================================================================
# MAIN — end-to-end pipeline (runs when you execute this file)
# ==================================================================

if __name__ == "__main__":
    baseline_df = pd.read_excel(DATASET_PATH, sheet_name="Baseline_M1_M3")
    all_df      = pd.read_excel(DATASET_PATH, sheet_name="All_Transactions")

    stats   = task1_baseline_stats(baseline_df)
    batches = task2_split_batches(all_df)

    alerts  = task3_mean_shift_detection(batches, stats)
    log     = task4_drift_log(batches, stats, alerts)

    ks_results = task5_ks_test(batches, baseline_df)
    comparison = task6_method_comparison(alerts, ks_results)

    model, scaler, acc, f1 = task7_train_baseline_model(baseline_df)
    performance = task8_evaluate_on_batches(model, scaler, batches)

    task9_drift_dashboard(batches, stats, alerts, ks_results, performance)

    print("\n--- Pipeline complete ---")
    print(f"Baseline model  Accuracy: {acc:.4f}   F1: {f1:.4f}")
    print(f"Mean-shift alerts : {[m for m, v in alerts.items() if v]}")
    print(f"KS-test alerts    : {[m for m, v in ks_results.items() if v['drifted']]}")