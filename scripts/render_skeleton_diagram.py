"""Render docs/images/keypoints_skeleton.{png,svg}.

Takes the AI-generated clean sedan background (`keypoints_skeleton_base.png`)
and overlays the 14 canonical CarFusion keypoints (naming from
dineshreddy91/Occlusion_Net/lib/data_loader/datasets/keypoint.py) and
the 18 skeleton edges shared with the production inference overlay.

Run: `.venv/Scripts/python.exe scripts/render_skeleton_diagram.py`
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import FancyBboxPatch

from vehicle_keypoints.inference.overlay import CARFUSION_SKELETON

REPO = Path(__file__).resolve().parents[1]
BASE_IMG = REPO / "docs" / "images" / "keypoints_skeleton_base.png"
OUT_PNG = REPO / "docs" / "images" / "keypoints_skeleton.png"
OUT_SVG = REPO / "docs" / "images" / "keypoints_skeleton.svg"

KEYPOINT_NAMES: tuple[str, ...] = (
    "Right Front wheel",
    "Left Front wheel",
    "Right Back wheel",
    "Left Back wheel",
    "Right Front HeadLight",
    "Left Front HeadLight",
    "Right Back HeadLight",
    "Left Back HeadLight",
    "Exhaust",
    "Right Front Top",
    "Left Front Top",
    "Right Back Top",
    "Left Back Top",
    "Center",
)

# Car in the base image faces LEFT (nose on the left, rear on the right).
# The left side of the car is closer to the viewer; the right side is occluded.
NEAR_SIDE = {1, 3, 5, 7, 8, 10, 12, 13}
FAR_SIDE = {0, 2, 4, 6, 9, 11}

# Keypoint pixel positions on the 1696x624 background (picked manually
# in scripts/tools/pick_keypoints.html).
POS: dict[int, tuple[int, int]] = {
    0: (362, 464),  # Right Front wheel
    1: (744, 457),  # Left Front wheel
    2: (1020, 453),  # Right Back wheel
    3: (1373, 439),  # Left Back wheel
    4: (237, 300),  # Right Front HeadLight
    5: (559, 309),  # Left Front HeadLight
    6: (1290, 215),  # Right Back HeadLight
    7: (1480, 224),  # Left Back HeadLight
    8: (1454, 458),  # Exhaust
    9: (701, 67),  # Right Front Top
    10: (956, 73),  # Left Front Top
    11: (1064, 40),  # Right Back Top
    12: (1200, 64),  # Left Back Top
    13: (940, 294),  # Center
}

TITLE = "CarFusion \u2014 14 anatomical keypoints + 18 skeleton edges"
BG = "#f7f8fa"
OUTLINE = "#1d2a3a"
KPT_FILL = "#1bb6d1"
EDGE_COLOR = "#e89a1b"
OCCLUDED_ALPHA = 0.45
DPI = 110


def render() -> None:
    img = imread(str(BASE_IMG))
    h_img, w_img = img.shape[:2]

    legend_w = int(w_img * 0.42)
    title_h = 110
    canvas_w = w_img + legend_w
    canvas_h = h_img + title_h

    fig = plt.figure(
        figsize=(canvas_w / DPI, canvas_h / DPI),
        dpi=DPI,
        facecolor=BG,
    )

    ax_img = fig.add_axes((0.0, 0.0, w_img / canvas_w, h_img / canvas_h))
    ax_img.imshow(img, extent=(0, w_img, h_img, 0))
    ax_img.set_xlim(0, w_img)
    ax_img.set_ylim(h_img, 0)
    ax_img.axis("off")

    fig.text(
        0.018,
        1 - 0.38 * title_h / canvas_h,
        TITLE,
        fontsize=22,
        color=OUTLINE,
        weight="bold",
        family="DejaVu Sans",
    )

    for a, b in CARFUSION_SKELETON:
        xa, ya = POS[a]
        xb, yb = POS[b]
        both_near = (a in NEAR_SIDE) and (b in NEAR_SIDE)
        alpha = 1.0 if both_near else 0.55
        ax_img.plot(
            [xa, xb],
            [ya, yb],
            color=EDGE_COLOR,
            linewidth=3,
            alpha=alpha,
            solid_capstyle="round",
            zorder=2,
        )

    for idx in range(14):
        x, y = POS[idx]
        near = idx in NEAR_SIDE
        alpha = 1.0 if near else OCCLUDED_ALPHA
        hollow = idx == 13
        linestyle = "--" if not near else "-"
        marker_size = 26**2
        if hollow:
            ax_img.scatter(
                [x],
                [y],
                s=marker_size,
                facecolors="white",
                edgecolors=OUTLINE,
                linewidths=2,
                alpha=alpha,
                zorder=3,
                linestyle=linestyle,
            )
        else:
            ax_img.scatter(
                [x],
                [y],
                s=marker_size,
                facecolors=KPT_FILL,
                edgecolors=OUTLINE,
                linewidths=2,
                alpha=alpha,
                zorder=3,
                linestyle=linestyle,
            )
        ax_img.text(
            x,
            y,
            str(idx),
            ha="center",
            va="center",
            color=OUTLINE if hollow else "white",
            fontsize=11,
            weight="bold",
            alpha=alpha,
            zorder=4,
        )

    ax_leg = fig.add_axes((w_img / canvas_w, 0.0, legend_w / canvas_w, h_img / canvas_h))
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")

    card = FancyBboxPatch(
        (0.04, 0.05),
        0.92,
        0.9,
        boxstyle="round,pad=0.005,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=OUTLINE,
        facecolor="white",
        transform=ax_leg.transAxes,
    )
    ax_leg.add_patch(card)

    ax_leg.text(
        0.5,
        0.02,
        "Dashed outline = occluded (far) side of the car",
        transform=ax_leg.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color=OUTLINE,
        style="italic",
        alpha=0.75,
    )

    row_h = 0.85 / 14
    top_row = 0.08 + 0.85 - row_h * 0.5
    for i, name in enumerate(KEYPOINT_NAMES):
        y_frac: float = top_row - row_h * i
        hollow = i == 13
        if hollow:
            ax_leg.scatter(
                [0.12],
                [y_frac],
                s=16**2,
                facecolors="none",
                edgecolors=OUTLINE,
                linewidths=1.5,
                transform=ax_leg.transAxes,
            )
        else:
            ax_leg.scatter(
                [0.12],
                [y_frac],
                s=16**2,
                facecolors=KPT_FILL,
                edgecolors=OUTLINE,
                linewidths=1.5,
                transform=ax_leg.transAxes,
            )
        ax_leg.text(
            0.18,
            y_frac,
            f"{i:>2}. {name}",
            transform=ax_leg.transAxes,
            ha="left",
            va="center",
            fontsize=11,
            color=OUTLINE,
            family="DejaVu Sans",
        )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=DPI, facecolor=BG)
    fig.savefig(OUT_SVG, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(REPO)}")
    print(f"wrote {OUT_SVG.relative_to(REPO)}")


if __name__ == "__main__":
    render()
