import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from GradientBooster.gradient_booster import GradientBoosterClassifier
except ImportError:
    from gradient_booster import GradientBoosterClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

EVAL_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_fig")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def compute_log_loss(y_true, y_proba, eps=1e-15):
    """Binary cross-entropy (log loss), computed from scratch.

        log loss = -mean( y*log(p) + (1-y)*log(1-p) )

    Average negative log-likelihood of true labels under predicted probabilities.
    Confident-correct ≈ 0, confident-wrong is heavily penalized. Probabilities 
    clipped to [eps, 1-eps] to avoid log(0) which is -inf.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted positive-class probabilities.
        eps: Clip bound keeping probabilities strictly inside (0, 1).

    Returns:
        Mean log loss; lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.clip(np.asarray(y_proba, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba)))


def get_scores(y_true, y_pred, y_proba=None):
    """Compute classification metrics for one model and return them as a dict.

    TP/FP/FN/TN, accuracy, precision, recall, and F1 are computed manually. 
    ROC-AUC uses sklearn's implementation since manual computation is 
    significantly more involved.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Hard predictions (0 or 1).
        y_proba: Optional positive-class probabilities; required for ROC-AUC.

    Returns:
        Dict with keys: tp, fp, fn, tn, accuracy, precision, recall, f1, and
        (when y_proba is given) roc_auc, pr_auc, log_loss.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    #count the four outcomes
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    
    #compute metrics from the counts
    out = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
    }
    
    out["f1"] = (
        2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])
        if (out["precision"] + out["recall"]) > 0 else 0.0
    )
    
    if y_proba is not None:
        # ROC-AUC: probability the model ranks a random positive above a random
        # negative; threshold-free
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
        out["pr_auc"] = average_precision_score(y_true, y_proba)
        out["log_loss"] = compute_log_loss(y_true, y_proba)

    return out


def plot_roc_pr(models, X_te, y_te, filename="roc_pr_all_quarters"):
    """
    Generate overlaid ROC and PR curves for multiple classifiers.
    Both curves are threshold-free comparisons.
    
    Args:
        models: Dict of display name -> trained classifier with predict_proba.
        X_te: Test features.
        y_te: Test labels.
        save_path: Output path for the saved PNG.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 5))
  
    for name, model in models.items():
          #class-1 probabilities: sklearn returns 2 columns, ours returns 1
          proba = model.predict_proba(X_te)
          if proba.ndim == 2:
              proba = proba[:, 1]
  
          #ROC: true-positive rate vs false-positive rate at every threshold
          fpr, tpr, _ = roc_curve(y_te, proba)
          roc_auc = roc_auc_score(y_te, proba)
          ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
  
          #PR: precision vs recall at every threshold (better for imbalance)
          prec, rec, _ = precision_recall_curve(y_te, proba)
          pr_auc = average_precision_score(y_te, proba)
          ax_pr.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})")
  
    # ROC: diagonal = random guessing
    ax_roc.plot([0, 1], [0, 1], "--", color="gray", label="chance (AUC = 0.5)")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR"); ax_roc.set_title("ROC curve")
    ax_roc.legend(loc="lower right")
  
    # PR: horizontal baseline = the positive-class rate (what a no-skill model gets)
    base = float(np.mean(y_te))
    ax_pr.axhline(base, ls="--", color="gray", label=f"baseline ({base:.2f})")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision"); ax_pr.set_title("PR curve")
    ax_pr.legend(loc="lower left")
  
    fig.tight_layout()
    os.makedirs(EVAL_FIG_DIR, exist_ok=True)
    out_path = os.path.join(EVAL_FIG_DIR, f"{filename}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(models, X_te, y_te, filename="confusion_matrix_all_quarters"):
    """Plot confusion matrices for multiple classifiers side-by-side.

    Args:
        models: Dict of display name -> trained classifier with .predict().
        X_te: Test features.
        y_te: Test labels.
        filename: Base filename.
    """
    class_names = ["0", "1"]
    # cell_labels[i][j] = category when true label = i, predicted = j
    cell_labels = [["True Negative", "False Positive"],
                   ["False Negative", "True Positive"]]

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_te, model.predict(X_te))
        total = cm.sum()
        im = ax.imshow(cm, cmap="Blues")

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
        ax.set_yticks([0, 1]); ax.set_yticklabels(class_names, rotation=90, va="center")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        # annotate each cell with category, count, and percentage
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = 100 * count / total if total else 0
                color = "white" if count > thresh else "black"
                ax.text(j, i, f"{cell_labels[i][j]}\n{count}\n({pct:.1f}%)",
                        ha="center", va="center", color=color, fontsize=11)

        fig.colorbar(im, ax=ax, shrink=0.75)

    fig.tight_layout()
    os.makedirs(EVAL_FIG_DIR, exist_ok=True)
    out_path = os.path.join(EVAL_FIG_DIR, f"{filename}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_correctness():
    """Is our GB implemented correctly? Train on a known deterministic rule; 
    high accuracy means boosting works."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(3000, 8)) #3000 samples, 8 features, values 0–10
    #label = 1 when feature 0 < 5 AND feature 1 > 5; 0 otherwise
    y = ((X[:, 0] < 5) & (X[:, 1] > 5)).astype(int) 

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size = 0.2, random_state = 0)
    model = GradientBoosterClassifier(n_trees= 50, learning_rate= 0.1, max_tree_depth = 3)
    model.fit(X_tr, y_tr)

    #only care about if the model can or cannot learn a known rule
    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"\naccuracy on y=(x0<5 AND x1>5): {acc:.4f}", "PASS" if acc > 0.95 else "FAIL")


def is_my_gb_good(X, y, params):
    """Is our GB good on real data? Compare it, on the same split, to
    sklearn's GB (ceiling). Good = near sklearn, judged by recall/F1/AUC
    (not accuracy)."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    ours = GradientBoosterClassifier(**params)
    ours.fit(X_tr, y_tr)
    ours_pred = ours.predict(X_te)
    #score our GB: hard predictions for accuracy/precision/recall, proba for auc
    ours_score = get_scores(y_te, ours_pred, ours.predict_proba(X_te))

    sk = GradientBoostingClassifier(n_estimators = params["n_trees"], learning_rate = params["learning_rate"], 
                                    max_depth = params["max_tree_depth"], random_state = 42).fit(X_tr, y_tr)
    sk_score = get_scores(y_te, sk.predict(X_te), sk.predict_proba(X_te)[:, 1])

    #print the comparison table
    print(f"\n{'metric':<12}{'our GB':>10}{'sklearn':>10}")
    for k in ["tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss"]:
        if k in ("tp", "fp", "fn", "tn"):
            o = ours_score.get(k, "—")
            s = sk_score.get(k, "—")
            print(f"{k:<12}{o:>10}{s:>10}")
        else:
            o = ours_score.get(k, float('nan'))
            s = sk_score.get(k, float('nan'))
            print(f"{k:<12}{o:>10.4f}{s:>10.4f}")

    print("our confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_te, ours_pred))

    models = {"our GB": ours, "sklearn": sk}
    plot_roc_pr(models, X_te, y_te)
    plot_confusion_matrices(models, X_te, y_te)


def load_data():
    """Load the real exoplanet dataset. X is features and y is 0/1."""
    df = pd.read_csv(os.path.join(DATA_DIR, "features_with_koi.csv"))
    df = df.dropna(subset=["label"])
    df = df.dropna() 
    
    y = df["label"].values    
    X = df.drop(columns=["label", "kep_id"]).values 

    return X, y


def main():
    """Run both evaluation layers: correctness first, then real data performance."""
    evaluate_correctness()
    X, y = load_data()
    is_my_gb_good(X, y, params=dict(n_trees=100, learning_rate=0.1, max_tree_depth=3))

if __name__ == "__main__":
    main()