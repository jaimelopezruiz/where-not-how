"""Overlay of every extracted cube trajectory in the table frame (C2.5)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots 1, 2, 3, 5 of the reference palette; showcase is neutral ink and dashed
COLOURS = {"straight": "#2a78d6", "curve": "#eb6834", "turn": "#1baf7a", "multi": "#e87ba4", "showcase": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0da"


def _draw(ax, ep, category, pose, contact, label=None):
    colour = COLOURS.get(category, MUTED)
    style = "--" if category == "showcase" else "-"
    ax.plot(pose[:, 0], pose[:, 1], style, color=colour, lw=0.8, alpha=0.55)
    pushed = np.where(contact[:, None], pose[:, :2], np.nan)
    ax.plot(pushed[:, 0], pushed[:, 1], style, color=colour, lw=2.0, label=label)
    ax.plot(*pose[0, :2], "o", ms=4, color=colour, mec="white", mew=0.8)       # start
    ax.plot(*pose[-1, :2], "s", ms=4, color=colour, mec="white", mew=0.8)      # end


def _style(ax, title):
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_overlay(episodes, path):
    """episodes: list of (id, category, cube_xy_yaw (M, 3), contact (M,)). Writes a PNG.

    Left: all trajectories, one colour per category. Right: one panel per category, so the overlay never
    has to be read by colour alone. Thick = box being pushed, thin = at rest; circle = start, square = end.
    """
    cats = [c for c in COLOURS if any(e[1] == c for e in episodes)]
    xy = np.concatenate([e[2][:, :2] for e in episodes])
    pad = 0.03
    xlim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ylim = (xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    fig = plt.figure(figsize=(14, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[2.2, 1, 1, 1])
    big = fig.add_subplot(gs[:, 0])
    for ep, cat, pose, contact in episodes:
        _draw(big, ep, cat, pose, contact)
    for c in cats:
        big.plot([], [], "--" if c == "showcase" else "-", color=COLOURS[c], lw=2,
                 label=f"{c} ({sum(e[1] == c for e in episodes)})")
    big.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    _style(big, f"All {len(episodes)} trajectories")
    for k, c in enumerate(cats):
        ax = fig.add_subplot(gs[k // 3, 1 + k % 3])
        for ep, cat, pose, contact in episodes:
            if cat == c:
                _draw(ax, ep, cat, pose, contact)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        _style(ax, f"{c} ({sum(e[1] == c for e in episodes)})")
    for ax in fig.axes:
        ax.set_xlabel("x (m)", fontsize=8, color=MUTED)
        ax.set_ylabel("y (m)", fontsize=8, color=MUTED)
    fig.suptitle("Cube trajectories in the table frame (board origin, z up)", fontsize=12, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
