import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

RISK_ORDER = ["Low", "Medium", "High", "Very High"]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_mae_hist(errors, out_dir):
    ensure_dir(out_dir)
    plt.figure(figsize=(8, 6))
    
    max_err = int(max(errors)) if errors else 5
    bins = np.arange(-0.5, max_err + 1.5, 1) 
    
    plt.hist(errors, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Absolute REBA Error (Points)")
    plt.ylabel("Frequency (Persons)")
    plt.title("Distribution of Errors (Lower is Better)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(range(max_err + 2))
    plt.savefig(os.path.join(out_dir, "mae_hist.png"))
    plt.close()

def plot_gt_vs_pred_scatter(gt_scores, pred_scores, out_dir):
    ensure_dir(out_dir)
    plt.figure(figsize=(7, 7))
    
    jitter_gt = np.array(gt_scores) + np.random.uniform(-0.2, 0.2, len(gt_scores))
    jitter_pred = np.array(pred_scores) + np.random.uniform(-0.2, 0.2, len(pred_scores))

    plt.scatter(jitter_gt, jitter_pred, alpha=0.5, color='royalblue', s=40)
    
    min_val = min(min(gt_scores), min(pred_scores))
    max_val = max(max(gt_scores), max(pred_scores))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Fit")

    plt.xlabel("Ground Truth REBA Score")
    plt.ylabel("Predicted REBA Score")
    plt.title("GT vs Predicted Scores (Points above line = Overestimation)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(out_dir, "gt_vs_pred_scatter.png"))
    plt.close()

def plot_risk_confusion_matrix(gt_levels, pred_levels, out_dir):
    ensure_dir(out_dir)
    cm = confusion_matrix(gt_levels, pred_levels, labels=RISK_ORDER)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=RISK_ORDER,
        yticklabels=RISK_ORDER,
        cmap="Blues",
        cbar=False,
        square=True
    )
    plt.xlabel("Predicted Risk Level")
    plt.ylabel("Actual Risk Level (Ground Truth)")
    plt.title("Confusion Matrix (Risk Classification)")
    plt.savefig(os.path.join(out_dir, "risk_confusion_matrix.png"))
    plt.close()

def plot_risk_bar_comparison(gt_levels, pred_levels, out_dir):
    ensure_dir(out_dir)
    
    gt_counts = {k: gt_levels.count(k) for k in RISK_ORDER}
    pred_counts = {k: pred_levels.count(k) for k in RISK_ORDER}

    x = np.arange(len(RISK_ORDER))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, [gt_counts[k] for k in RISK_ORDER], width, label='Ground Truth', color='gray')
    plt.bar(x + width/2, [pred_counts[k] for k in RISK_ORDER], width, label='Predicted', color='cornflowerblue')

    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.title('Risk Level Distribution Comparison')
    plt.xticks(x, RISK_ORDER)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "risk_level_comparison.png"))
    plt.close()

def plot_error_vs_confidence(errors, confidences, out_dir):
    ensure_dir(out_dir)

    plt.figure(figsize=(8, 5))
    plt.scatter(confidences, errors, alpha=0.6, color='purple')
    plt.xlabel("Model Confidence")
    plt.ylabel("Absolute Error")
    plt.title("Error vs Confidence")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(out_dir, "error_vs_confidence.png"))
    plt.close()

def generate_plots(results, out_dir):
    # 1. Error Distribution
    plot_mae_hist(results["errors"], out_dir)
    
    # 2. Scatter Correlation (New & Better)
    plot_gt_vs_pred_scatter(results["gt_scores"], results["pred_scores"], out_dir)
    
    # 3. The Star of the Show: Confusion Matrix
    plot_risk_confusion_matrix(results["gt_levels"], results["pred_levels"], out_dir)
    
    # 4. High Level Overview
    plot_risk_bar_comparison(results["gt_levels"], results["pred_levels"], out_dir)
    
    # 5. Diagnostic
    plot_error_vs_confidence(results["errors"], results["confidences"], out_dir)