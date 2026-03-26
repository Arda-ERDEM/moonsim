from __future__ import annotations

from pathlib import Path

import numpy as np

from moon_gen.planning import config
from moon_gen.planning.types import (
    LayerMap,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency at runtime
    plt = None


def _plot_path(ax, path: list[tuple[int, int]], color: str, label: str):
    if not path:
        return
    rr = np.array([p[0] for p in path])
    cc = np.array([p[1] for p in path])
    ax.plot(cc, rr, color=color, linewidth=2.0, label=label)


def render_outputs(
    image: np.ndarray,
    layers: LayerMap,
    cost_maps: dict[str, np.ndarray],
    plans: dict[str, dict],
    selected_mode: str | None,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> dict[str, str]:
    outputs: dict[str, str] = {}

    if not config.VISUALIZE:
        return outputs
    if plt is None:
        print("[Visualization] matplotlib is not available; skipping figures.")
        return outputs

    out_dir = Path(config.VISUALIZE_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. OVERVIEW FIGURE
    fig1, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=config.VISUALIZE_DPI)
    fig1.suptitle("Lunar Multi-Layer Terrain Analysis")

    views = [
        ("Original image", image, "gray"),
        ("Slope [approx from image gradient]", layers["slope"], "viridis"),
        ("Roughness", layers["roughness"], "magma"),
        ("Obstacle risk", layers["obstacle"], "gray_r"),
        ("Crater risk (proxy)", layers["crater"], "inferno"),
        ("Confidence", layers["confidence"], "cividis"),
        ("Safe cost map", cost_maps["safe"], "plasma"),
        ("Eco cost map", cost_maps["eco"], "plasma"),
        ("Fast cost map", cost_maps["fast"], "plasma"),
    ]

    for ax, (title, arr, cmap) in zip(axes.ravel(), views):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.scatter([start[1], goal[1]], [start[0], goal[0]], c=["lime", "red"], s=20)
        ax.set_xticks([])
        ax.set_yticks([])
        fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    overview_path = out_dir / config.OVERVIEW_FIGURE_NAME
    if config.VISUALIZE_SAVE:
        fig1.tight_layout()
        fig1.savefig(overview_path)
        outputs["overview"] = str(overview_path)

    # 2. SELECTION FIGURE
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11), dpi=config.VISUALIZE_DPI)
    # top-left safe, top-right eco, bottom-left fast, bottom-right selected
    panels = [
        ("safe", axes2[0, 0]),
        ("eco", axes2[0, 1]),
        ("fast", axes2[1, 0]),
    ]

    for mode, ax in panels:
        ax.imshow(cost_maps[mode], cmap="viridis")
        path = plans[mode]["result"].smoothed_path
        color = config.MODE_COLORS.get(mode, "white")
        _plot_path(ax, path, color, f"{mode.upper()} path")
        ax.scatter([start[1], goal[1]], [start[0], goal[0]], c=["lime", "red"], s=35)
        ax.set_title(f"{mode.upper()} mode")
        ax.legend(loc="upper right")
        ax.set_xticks([])
        ax.set_yticks([])

    # Final selection panel
    final_ax = axes2[1, 1]
    final_ax.imshow(image, cmap="gray")
    final_ax.scatter([start[1], goal[1]], [start[0], goal[0]], c=["lime", "red"], s=40)
    if selected_mode is not None:
        selected_path = plans[selected_mode]["result"].smoothed_path
        color = config.MODE_COLORS.get(selected_mode, "white")
        _plot_path(final_ax, selected_path, color, f"Selected: {selected_mode.upper()}")
        final_ax.legend(loc="upper right")
        final_ax.set_title("Autonomous final selection")
    else:
        final_ax.set_title("Autonomous final selection (none)")
    final_ax.set_xticks([])
    final_ax.set_yticks([])

    selection_path = out_dir / config.SELECTION_FIGURE_NAME
    if config.VISUALIZE_SAVE:
        fig2.tight_layout()
        fig2.savefig(selection_path)
        outputs["selection"] = str(selection_path)

    if config.VISUALIZE_SHOW:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

    return outputs
