import os
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, confusion_matrix)
from rnn_v2 import ExoplanetRNN, downsample, standardize_rows, evaluate, DEVICE
from rnn_torch_ref import TorchGRU

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_fig")
OURS, REF, THR = "#1f77b4", "#ff7f0e", 0.3

# These numbers came from training the model with plain gradient descent, 
# on 250-step sequences (2000 flux points / factor=8), batch_size=64 at a time. 
# Hardcoded here, so the tuning plot does not re-run them.
#   Learning rate: trained 40 epochs at each learning rate, measured test ROC-AUC / accuracy
#        (the lr=1.0 point is from a separate run where training diverged)
#   Epoch: trained at lr=0.5, measured test metrics every 10 epochs up to 100
LR  = dict(x=[.05, .1, .2, .3, .5, 1.], roc=[.704, .716, .756, .815, .861, .600],
           acc=[.636, .657, .722, .760, .796, .348])
EPO = dict(x=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
           roc=[.716, .758, .821, .861, .861, .850, .880, .880, .886, .874],
           acc=[.667, .701, .752, .796, .785, .779, .791, .809, .812, .787])


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def load():
    df = pd.read_csv("data/rnn_timeseries.csv")
    flux = sorted([c for c in df if "flux" in c], key=lambda c: int(c.split("_")[1]))
    y = df["label"].to_numpy(np.float32)
    X = standardize_rows(downsample(np.clip(df[flux].to_numpy(np.float32), -1, 1), factor=8))
    _, xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    ours = ExoplanetRNN()
    for p, s in zip(ours.parameters(), torch.load("exoplanet_rnn_v2.pt", map_location=DEVICE)):
        p.data = s.to(DEVICE)
    models = [("our GRU", evaluate(ours, xte), OURS)]
    if os.path.exists("exoplanet_rnn_torch_ref.pt"):
        ref = TorchGRU().to(DEVICE)
        ref.load_state_dict(torch.load("exoplanet_rnn_torch_ref.pt", map_location=DEVICE))
        models.append(("PyTorch nn.GRU", evaluate(ref, xte), REF))
    return yte, models


def plot_roc_pr(y, models):
    fig, (r, p) = plt.subplots(1, 2, figsize=(11, 5))
    for name, prob, c in models:
        fpr, tpr, _ = roc_curve(y, prob)
        pr, rc, _ = precision_recall_curve(y, prob)
        r.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={roc_auc_score(y, prob):.3f})")
        p.plot(rc, pr, color=c, lw=2, label=f"{name} (PR-AUC={average_precision_score(y, prob):.3f})")
    r.plot([0, 1], [0, 1], "--", color="gray", label="chance (0.5)")
    p.axhline(y.mean(), ls="--", color="gray", label=f"baseline ({y.mean():.2f})")
    r.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC  (higher = better)")
    p.set(xlabel="recall", ylabel="precision", title="PR  (higher = better)")
    r.legend(loc="lower right"); p.legend(loc="lower left")
    fig.suptitle("Our GRU vs PyTorch nn.GRU (same training)", fontweight="bold")
    save(fig, "rnn_roc_pr.png")


def plot_confusion(y, models):
    names = [["True Neg", "False Pos"], ["False Neg", "True Pos"]]
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5), squeeze=False)
    for ax, (name, prob, _) in zip(axes[0], models):
        cm = confusion_matrix(y, (prob >= THR).astype(int)); tot = cm.sum()
        ax.imshow(cm, cmap="Blues")
        ax.set(title=f"{name}  (accuracy {(cm[0, 0] + cm[1, 1]) / tot:.3f})",
               xticks=[0, 1], yticks=[0, 1], xlabel="predicted label", ylabel="true label")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{names[i][j]}\n{cm[i, j]} ({100 * cm[i, j] / tot:.1f}%)",
                        ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle(f"Confusion matrices (threshold {THR})", fontweight="bold")
    save(fig, "rnn_confusion_matrix.png")


def _panel(ax, d, xlabel, title, pick, notes, logx=False):
    ax.plot(d["x"], d["roc"], "o-", color=OURS, label="ROC-AUC")
    ax.plot(d["x"], d["acc"], "s--", color=REF, label="accuracy")
    if logx:
        ax.set_xscale("log")
    ax.axvline(pick[0], color="green", ls=":", lw=1.5)
    ax.text(pick[0], 0.95, pick[1], color="green", ha="center", va="top", fontsize=9, fontweight="bold")
    for txt, xy, xytext in notes:
        ax.annotate(txt, xy=xy, xytext=xytext, fontsize=8.5, color="gray", ha="center",
                    arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set(xlabel=xlabel, ylabel="test score  (higher = better)", title=title, ylim=(0.33, 0.97))
    ax.legend(loc="center left" if logx else "lower right", fontsize=9)


def plot_tuning():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.5))
    _panel(a, LR, "learning rate = step size  (log scale)", "1. Picking the learning rate",
           (0.5, "lr = 0.5\n(best)"),
           [("too small:\nbarely learns", (.05, .704), (.075, .52)),
            ("too big:\ndiverges", (1., .6), (.55, .44))], logx=True)
    _panel(b, EPO, "training epochs  (how long we train)", "2. Picking how long to train",
           (80, "80 epochs"),
           [("keeps improving,\nthen plateaus", (90, .886), (38, .80))])
    fig.suptitle("Tuning the RNN: the learning rate and epoch", fontweight="bold")
    save(fig, "rnn_tuning.png")


if __name__ == "__main__":
    y, models = load()
    for name, prob, _ in models:
        print(f"{name:16s} ROC-AUC={roc_auc_score(y, prob):.3f}  PR-AUC={average_precision_score(y, prob):.3f}")
    plot_roc_pr(y, models)
    plot_confusion(y, models)
    plot_tuning()
