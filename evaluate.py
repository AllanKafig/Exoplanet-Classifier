import numpy as np
import pandas as pd
import matplotlib as plt
from gradient_booster import GradientBoosterClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def get_scores(y_true, y_pred, y_proba=None):
    """Bundle the classification metrics for one model into a dict, 
    so our GB and sklearn's GB are all scored the same way."""
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
        #roc_auc uses the probabilities (not the 0/1 labels) to score how well the
        #model ranks planets above non-planets;
        #not too sure on how to get auc, used sklearn's implementation
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    
    return out

def plot_learning_curve(X, y, save_path="learning_curve.png"):
    """Plot how F1 and ROC-AUC evolve as n_trees increases.
    Justifies the choice of n_trees by showing where performance plateaus."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size = 0.2, random_state = 42, stratify = y
    )
    
    n_trees_list = [10, 20, 40, 60, 100]
    f1s = []
    aucs = []
    
    for n in n_trees_list:
        print(f"Training with n_trees={n}")
        model = GradientBoosterClassifier(n_trees=n, learning_rate=0.1, max_tree_depth=3)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)
        if proba.ndim == 2:
            proba = proba[:, 1]
       
        scores = get_scores(y_te, pred, proba)
        f1s.append(scores["f1"])
        aucs.append(scores["roc_auc"])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_trees_list, f1s, "o-", label="F1", color="#3266ad", linewidth=2)
    ax.plot(n_trees_list, aucs, "s-", label="ROC-AUC", color="#73726c", linewidth=2)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Score")
    ax.set_title("Performance vs number of trees (custom GB)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def evaluate_correctness():
    """is our GB implemented correctly? Train on a known deterministic rule; 
    high accuracy means boosting works."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(3000, 8)) #3000 samples, 8 features, values 0–10
    # label = 1 when feature 0 < 5 AND feature 1 > 5; 0 otherwise
    y = ((X[:, 0] < 5) & (X[:, 1] > 5)).astype(int) 

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size = 0.2, random_state = 0)
    model = GradientBoosterClassifier(n_trees= 50, learning_rate= 0.1, max_tree_depth = 3)
    model.fit(X_tr, y_tr)

    #only care about if the model can or cannot learn a known rule
    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"\naccuracy on y=(x0<5 AND x1>5): {acc:.4f}", "PASS" if acc > 0.95 else "FAIL")


def is_my_gb_good(X, y, params):
    """is our GB good on real data? Compare it, on the same split, to
    sklearn's GB (ceiling) and an always-majority dummy (floor). Good = near
    sklearn and well above dummy, judged by recall/F1/AUC (not accuracy)."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    ours = GradientBoosterClassifier(**params)
    ours.fit(X_tr, y_tr)
    ours_pred = ours.predict(X_te)
    #score our GB: hard predictions for accuracy/precision/recall, proba for auc
    ours_score = get_scores(y_te, ours_pred, ours.predict_proba(X_te))

    sk = GradientBoostingClassifier(n_estimators = params["n_trees"], learning_rate = params["learning_rate"], 
                                    max_depth = params["max_tree_depth"], random_state = 42).fit(X_tr, y_tr)
    sk_score = get_scores(y_te, sk.predict(X_te), sk.predict_proba(X_te)[:, 1])

    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    dummy_score = get_scores(y_te, dummy.predict(X_te), dummy.predict_proba(X_te)[:, 1])
    
    #print the comparison table 
    print(f"\n{'metric':<12}{'our GB':>10}{'sklearn':>10}{'dummy':>10}")
    for k in ["tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1", "roc_auc"]:
        if k in ("tp", "fp", "fn", "tn"):
            o = ours_score.get(k, "—")
            s = sk_score.get(k, "—")
            d = dummy_score.get(k, "—")
            print(f"{k:<12}{o:>10}{s:>10}{d:>10}")
        else:
            o = ours_score.get(k, float('nan'))
            s = sk_score.get(k, float('nan'))
            d = dummy_score.get(k, float('nan'))
            print(f"{k:<12}{o:>10.4f}{s:>10.4f}{d:>10.4f}")
        
    print("our confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_te, ours_pred))

def load_data():
    """Load the real exoplanet dataset. X is features and y is 0/1."""
    df = pd.read_csv("features.csv")
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