"""Data exploration plots for the RNN.

Views of the folded light-curve dataset / KOI catalog, each a function:
  - plot_folded_gallery   : zoomed folded-transit examples (grid, deep -> shallow per class)
  - plot_median_overlay   : depth-normalized median folded curve per class (+ percentile band)
  - plot_stellar_hosting  : host fraction vs. metallicity, and across the H-R diagram
  - plot_planet_types     : period-radius diagram (drawn planets) + transit depth by type

Run as a script to regenerate all of them, or import and call any one.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

HERE = os.path.dirname(os.path.abspath(__file__))     # .../src/data_exploration
SRC_DIR = os.path.join(HERE, "..")                    # .../src
DATA_DIR = os.path.join(HERE, "..", "..", "data")     # repo-root /data
FIG_DIR = HERE                                        # figures live alongside this script
sys.path.append(SRC_DIR)
from build_gb_dataset import get_star_labels   # CONFIRMED=1 / FP=0, CANDIDATE dropped

CONFIRMED_COLOR = "#1e88e5"
FP_COLOR = "#d1495b"

# radius-based planet types (Earth radii): (name, lo, hi, color)
PLANET_TYPES = [
    ("Earth-size",   0.0,  1.25, "#4daf4a"),
    ("super-Earth",  1.25, 2.0,  "#377eb8"),
    ("sub-Neptune",  2.0,  4.0,  "#984ea3"),
    ("Neptune",      4.0,  6.0,  "#ff7f00"),
    ("Jovian",       6.0,  40.0, "#e41a1c"),
]

# direction (dx, dy) to place each type's name relative to its glyph, chosen to
# spread the crowded central labels apart so they don't overlap
LABEL_DIR = {
    "Earth-size": (0, -1), "super-Earth": (-1, -0.3), "sub-Neptune": (1, -0.3),
    "Neptune": (1, 0.3), "Jovian": (0, 1),
}


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


def _planet_rgba(color, n=220):
    """Render a shaded sphere (light from upper-left) in `color` as an RGBA image."""
    yy, xx = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    r = np.hypot(xx, yy)
    inside = r <= 1.0
    nz = np.sqrt(np.clip(1 - r**2, 0, 1))
    light = np.clip(0.30 + 0.70 * (-0.45 * xx + 0.45 * yy + 0.75 * nz), 0, 1)
    rgb = np.array(to_rgb(color))
    img = np.zeros((n, n, 4))
    img[..., :3] = rgb[None, None, :] * light[..., None]
    img[..., 3] = inside.astype(float)
    return img


def plot_planet_types(F, y, kep):
    """How planet TYPES differ. (A) period-radius diagram with a drawn planet per
    radius-based type (sized by radius) over the confirmed-host population; (B)
    transit depth by type, from the folded curves. Catalog koi_prad/koi_period are
    used only as plot axes / for the radius classification."""
    meas_depth = -np.sort(F, axis=1)[:, :20].mean(axis=1)
    conf = (y == 1) & (meas_depth > 0)
    rnn = pd.DataFrame({"kep_id": kep[conf], "meas_depth": meas_depth[conf]})
    koi = (pd.read_csv(os.path.join(DATA_DIR, "koi_cumulative.csv"), comment="#")
           .drop_duplicates("kepid"))
    df = rnn.merge(koi[["kepid", "koi_prad", "koi_period"]], left_on="kep_id",
                   right_on="kepid", how="left").dropna(subset=["koi_prad", "koi_period"])
    df["type"] = pd.NA
    for name, lo, hi, _ in PLANET_TYPES:
        df.loc[(df.koi_prad >= lo) & (df.koi_prad < hi), "type"] = name

    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 7))
    handles = []   # legend carries the radius/count detail; glyphs get short names
    for name, lo, hi, color in PLANET_TYPES:
        s = df[df.type == name]
        if not len(s):
            continue
        a.scatter(s.koi_period, s.koi_prad, s=7, alpha=0.20, color=color)  # dot darkness (Lower -> fainter)
        med_p, med_r = np.median(s.koi_period), np.median(s.koi_prad)
        zoom = 0.06 + 0.03 * np.sqrt(med_r)              # bigger glyph for bigger planet
        a.add_artist(AnnotationBbox(OffsetImage(_planet_rgba(color), zoom=zoom),
                                    (med_p, med_r), frameon=False, zorder=5))
        # type name placed away from the cluster, white halo so it stands out
        gr = zoom * 120                                  # glyph radius in points
        dx, dy = LABEL_DIR.get(name, (0, -1))
        ha = "left" if dx > 0 else "right" if dx < 0 else "center"
        va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
        a.annotate(name, (med_p, med_r), xytext=(dx * (gr + 14), dy * (gr + 14)),
                   textcoords="offset points", ha=ha, va=va, fontsize=9.5,
                   color=color, fontweight="bold",
                   path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                              markersize=9, label=f"{name}  (~{med_r:.1f} R⊕, n={len(s)})"))
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("orbital period (days)")
    a.set_ylabel("planet radius (R⊕)")
    a.set_title("Where each planet type lives (drawn to scale by radius)")
    a.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    order = [t[0] for t in PLANET_TYPES]
    data = [df[df.type == name].meas_depth.values * 100 for name in order]
    bp = b.boxplot(data, patch_artist=True, showfliers=False, medianprops=dict(color="black"))
    for patch, t in zip(bp["boxes"], PLANET_TYPES):
        patch.set_facecolor(t[3]); patch.set_alpha(0.6)
    b.set_yscale("log")
    b.set_xticks(range(1, len(order) + 1))
    b.set_xticklabels(order, rotation=20, ha="right", fontsize=9)
    b.set_ylabel("measured transit depth (%)  [log]")
    b.set_title("Bigger planets make deeper transits (from folded curves)")

    fig.suptitle("How exoplanet types differ: radius classes (confirmed hosts)", fontsize=13)
    fig.text(0.5, 0.005, "R⊕ = Earth radii (planet size relative to Earth)",
             ha="center", fontsize=8, style="italic", color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    _save(fig, "planet_types.png")
    plt.close(fig)


def main():
    F, y, kep, phase = load_folded()
    plot_folded_gallery(F, y, kep, phase)
    plot_median_overlay(F, y, phase)
    plot_stellar_hosting()
    plot_planet_types(F, y, kep)


if __name__ == "__main__":
    main()
