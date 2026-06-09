"""Data exploration plots for the RNN.

Three views of the folded light-curve dataset / KOI catalog, each a function:
  - plot_folded_gallery   : zoomed folded-transit examples (grid, deep -> shallow per class)
  - plot_median_overlay   : depth-normalized median folded curve per class (+ percentile band)
  - plot_stellar_hosting  : host fraction vs. metallicity, and across the H-R diagram
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
FIG_DIR = os.path.join(HERE, "..", "data_exploration_plots")
sys.path.append(HERE)
from src.build_gb_dataset import get_star_labels   # CONFIRMED=1 / FP=0, CANDIDATE dropped

CONFIRMED_COLOR = "#1e88e5"
FP_COLOR = "#d1495b"


def load_folded():
    """Load the folded RNN dataset. Returns (F, y, kep, phase):
    F = flux matrix (n_stars x n_bins), y = labels, kep = Kepler IDs, phase axis."""
    df = pd.read_csv(os.path.join(DATA_DIR, "rnn_timeseries.csv"))
    flux = sorted([c for c in df.columns if c.startswith("flux_")],
                  key=lambda c: int(c.split("_")[1]))
    F = df[flux].to_numpy()
    y = df["label"].to_numpy()
    kep = df["kep_id"].to_numpy()
    phase = np.linspace(-0.5, 0.5, len(flux))
    return F, y, kep, phase


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", os.path.relpath(out, os.path.join(HERE, "..")))


def _span_by_depth(pool, depth, n):
    """Pick n indices from `pool`, spanning deepest->shallowest transit."""
    idx = pool[np.argsort(depth[pool])]          # deepest (most negative) first
    return [idx[int(f * (len(idx) - 1))] for f in np.linspace(0, 0.9, n)]


def _draw_transit_panel(ax, phase, fc, color, title, show_x, show_y):
    """Draw one auto-zoomed folded transit (raw + smoothed) on `ax`."""
    central = np.abs(phase) < 0.2
    d = fc[central].min()
    mask = central & (fc < 0.3 * d)
    half = np.abs(phase[mask]).max() if mask.sum() >= 2 else 0.01
    w = min(max(3 * half, 0.015), 0.5)

    ax.plot(phase, fc, color=color, lw=0.5, alpha=0.3)
    ax.plot(phase, np.convolve(fc, np.ones(7) / 7, mode="same"), color=color, lw=1.4)
    ax.axhline(0, color="gray", lw=0.4, ls=":")
    ax.axvline(0, color="gray", lw=0.4, ls=":")
    ax.set_xlim(-w, w)
    vis = np.abs(phase) <= w
    lo, hi = fc[vis].min(), fc[vis].max()
    pad = 0.12 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title, fontsize=7.5)
    ax.tick_params(labelsize=6)
    if show_y:
        ax.set_ylabel("norm. flux", fontsize=7)
    if show_x:
        ax.set_xlabel("orbital phase (zoom)", fontsize=7)


def plot_folded_gallery(F, y, kep, phase, cols=6, rows_per_class=2):
    """Grid of zoomed folded transits, deep->shallow per class (top: confirmed,
    bottom: false positives). Each panel auto-zooms to its own transit width."""
    n = cols * rows_per_class
    depth = F.min(axis=1)
    physical = depth > -1.0                 # drop the 2 unphysical (> 100% dimming) curves
    nrows = 2 * rows_per_class

    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 2.6, nrows * 2.1))
    for b, (label, color) in enumerate([(1, CONFIRMED_COLOR), (0, FP_COLOR)]):
        pool = np.where((y == label) & physical)[0]
        for k, i in enumerate(_span_by_depth(pool, depth, n)):
            r = b * rows_per_class + k // cols
            c = k % cols
            _draw_transit_panel(axes[r, c], phase, F[i], color,
                                f"KIC {int(kep[i])}\n{depth[i]*100:.2f}%",
                                show_x=(r == nrows - 1), show_y=(c == 0))

    fig.suptitle("Folded transits, zoomed — what the RNN reads  "
                 f"(top {n}: confirmed hosts;  bottom {n}: false positives;  each block spans deepest→shallowest)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "folded_examples_gallery.png")
    plt.close(fig)


def plot_median_overlay(F, y, phase):
    """Depth-normalized median folded curve per class with a 25-75% band. Scaling
    each curve to its own depth first keeps the central dip from washing out, and
    surfaces the false-positive secondary eclipse at phase +/-0.25."""
    Fc = np.clip(F, -1.0, 1.0)
    sm = uniform_filter1d(Fc, size=21, axis=1, mode="nearest")
    norm = sm / (-sm.min(axis=1, keepdims=True) + 1e-6)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, color, name in [(1, CONFIRMED_COLOR, "confirmed host"), (0, FP_COLOR, "false positive")]:
        M = norm[y == label]
        med = np.median(M, axis=0)
        lo, hi = np.percentile(M, [25, 75], axis=0)
        ax.fill_between(phase, lo, hi, color=color, alpha=0.18)
        ax.plot(phase, med, color=color, lw=1.8, label=f"{name} (n={len(M)})")

    for x in (-0.25, 0.25):
        ax.axvline(x, color="gray", ls="--", lw=0.8)
    ax.annotate("secondary eclipse\n(eclipsing binaries)", xy=(0.25, -0.15), xytext=(0.32, -0.45),
                fontsize=9, ha="center", arrowprops=dict(arrowstyle="->", color="gray"))
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("orbital phase")
    ax.set_ylabel("per-star depth-normalized flux  (median; band = 25–75%)")
    ax.set_title("Depth-normalized median folded curve by class — all 6640 stars")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "folded_median_overlay.png")
    plt.close(fig)


def plot_stellar_hosting():
    """Host fraction vs. metallicity (planet-metallicity correlation) and across
    the H-R diagram. NOTE: host fraction = confirmed/(confirmed+FP) among KOIs —
    a confirmation fraction, not a true occurrence rate."""
    labels = get_star_labels()
    koi = (pd.read_csv(os.path.join(DATA_DIR, "koi_cumulative.csv"), comment="#")
           .drop_duplicates("kepid"))
    df = labels.merge(koi[["kepid", "koi_smet", "koi_steff", "koi_slogg"]], on="kepid", how="left")
    base = df.label.mean()

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.6))

    m = df.dropna(subset=["koi_smet"])
    edges = np.linspace(m.koi_smet.quantile(0.02), m.koi_smet.quantile(0.98), 11)
    m = m.assign(mbin=pd.cut(m.koi_smet, edges))
    g = m.groupby("mbin", observed=True)["label"].agg(["mean", "count"])
    cx = np.array([iv.mid for iv in g.index])
    err = np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"])
    a.errorbar(cx, g["mean"], yerr=err, fmt="o-", color=CONFIRMED_COLOR, capsize=3)
    a.axhline(base, ls="--", color="gray", label=f"overall host rate ({base:.2f})")
    a.set_xlabel("stellar metallicity  [Fe/H]")
    a.set_ylabel("fraction confirmed host")
    a.set_title("Host fraction vs. metallicity")
    a.legend(loc="upper left")

    h = df.dropna(subset=["koi_steff", "koi_slogg"])
    hb = b.hexbin(h.koi_steff, h.koi_slogg, C=h.label, reduce_C_function=np.mean,
                  gridsize=28, cmap="coolwarm", mincnt=5)
    b.invert_xaxis()
    b.invert_yaxis()
    b.set_xlabel("effective temperature  koi_steff (K)")
    b.set_ylabel("surface gravity  koi_slogg (log g)")
    b.set_title("Host fraction across the H-R diagram")
    cb = fig.colorbar(hb, ax=b)
    cb.set_label("host fraction (mean label, ≥5 stars/cell)")

    fig.suptitle("Which stars host planets — by metallicity and stellar type", fontsize=13)
    fig.text(0.5, 0.01,
             "host fraction = confirmed/(confirmed+FP) among KOIs; not a true occurrence rate",
             ha="center", fontsize=8, style="italic", color="gray")
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    _save(fig, "stellar_hosting.png")
    plt.close(fig)


def main():
    F, y, kep, phase = load_folded()
    plot_folded_gallery(F, y, kep, phase)
    plot_median_overlay(F, y, phase)
    plot_stellar_hosting()


if __name__ == "__main__":
    main()
