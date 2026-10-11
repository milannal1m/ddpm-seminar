"""
DDPM-style forward diffusion (noising) process visualization.

Adds Gaussian noise to an image following the forward process of a
Denoising Diffusion Probabilistic Model (DDPM, Ho et al. 2020):

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

which has the closed-form sampling equation:

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon,  epsilon ~ N(0, I)

Produces a single figure with the image at 5 points along the noising
schedule, suitable for use in a paper.
"""

import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
# Every label is typeset by a real LaTeX run when latex + dvipng are on PATH,
# so the figure notation is not an approximation of the paper's typography -
# it is the same typesetter producing it. This also matters for correctness:
# matplotlib's built-in Computer Modern has no bold digits, so \mathbf{0}
# silently renders light-weight there, while LaTeX sets it properly bold.
#
# Falls back to matplotlib's own mathtext (Computer Modern) if TeX is missing,
# so the script still runs on a machine without a LaTeX install.
USE_LATEX = shutil.which("latex") is not None and shutil.which("dvipng") is not None

plt.rcParams["font.family"] = "serif"
if USE_LATEX:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amssymb,bm}"
else:
    # "cm" = Computer Modern, LaTeX's default; use "stix" for a Times-based
    # template (ACM / IEEE / Springer).
    MATHTEXT_FONTSET = "cm"
    _TEXT_FONTS = {
        "cm": ["cmr10", "DejaVu Serif"],  # cmr10 ships with matplotlib
        "stix": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    }
    plt.rcParams["mathtext.fontset"] = MATHTEXT_FONTSET
    plt.rcParams["font.serif"] = _TEXT_FONTS.get(MATHTEXT_FONTSET, ["DejaVu Serif"])
    plt.rcParams["axes.unicode_minus"] = False   # cmr10 has no U+2212 glyph
    plt.rcParams["axes.formatter.use_mathtext"] = True

# Vector notation, written exactly as in the paper's equations:
#   \mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon},
#   \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
# i.e. upright bold for the image/matrix symbols, bold italic for the Greek noise
# symbol - the standard convention, and what the prose uses.
VEC_X = r"\mathbf{x}"
VEC_EPS = r"\boldsymbol{\epsilon}"
VEC_ZERO = r"\mathbf{0}"
VEC_I = r"\mathbf{I}"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# All inputs and outputs live in images/ at the repo root, independent of cwd.
IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
IMAGE_PATH = IMAGES_DIR / "unnoised.jpg"
OUTPUT_PATH = IMAGES_DIR / "noising_steps.png"              # image row + pixel-density row
OUTPUT_PATH_PLAIN = IMAGES_DIR / "noising_steps_plain.png"  # image row only
DIAGRAM_OUTPUT_PATH = IMAGES_DIR / "diffusion_diagram.png"
EQUATION_OUTPUT_PATH = IMAGES_DIR / "equation_diagram.png"

T = 1000            # total diffusion steps in the schedule
BETA_START = 1e-4   # standard DDPM linear schedule
BETA_END = 0.02
N_STEPS_SHOWN = 5   # number of snapshots in the noising_steps.png figure
DIAGRAM_TIMESTEPS = [0, 150, 500, T - 1]  # x_0 (original), x_{t-1}, x_t, x_T (fully noised) - spread for visible contrast
EQUATION_TIMESTEP = 300  # which t to illustrate in the closed-form equation diagram
OUTPUT_PADDING_PX = 10   # whitespace border around each saved PNG, in pixels (not inches)
SEED = 42           # for reproducibility

# Pixel-histogram row under the noising_steps.png images.
# HIST_USE_CLIPPED picks what "the pixel distribution" means:
#   True  - the values actually displayed, i.e. clamped to [0, 1]. Matches the
#           image above it exactly, but the clamp piles all out-of-range mass
#           onto the two end bins, which dominates the late-t panels.
#   False - the raw x_t values. Shows the real Gaussian that the forward
#           process converges to, at the cost of covering a wider range than
#           the image above can display.
HIST_USE_CLIPPED = False
HIST_BINS = 80
HIST_TITLE_FS = 15
HIST_TICK_FS = 12
HIST_LABEL_FS = 13


# ---------------------------------------------------------------------------
# Data range
# ---------------------------------------------------------------------------
# The diffusion process operates on data scaled to [-1, 1], following Ho et al.
# and matching the paper's Section 3.1. This is not cosmetic: the forward
# process drives x_t towards N(0, I), and only a zero-centred x_0 of roughly
# unit scale sits inside that target. Data on [0, 1] would carry a mean of ~0.5
# that the schedule has to erase on top of destroying the image.
#
# Pixels are read as [0, 1] (that is what a file gives us) and converted at the
# two boundaries: to_model() once on load, to_display() whenever an array is
# handed to imshow. Everything in between - the closed form, the clamping, the
# histograms - is in model space, so the numbers on the histogram axis are the
# numbers the paper talks about.
def to_model(x01):
    """[0, 1] pixel values -> the [-1, 1] range the diffusion process uses."""
    return x01 * 2.0 - 1.0


def to_display(x):
    """[-1, 1] model values -> the [0, 1] range imshow expects, clamped."""
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Diffusion schedule
# ---------------------------------------------------------------------------
def make_schedule(T, beta_start, beta_end):
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return alpha_bars


def forward_noise_raw(x0, alpha_bar_t, rng):
    """
    Sample x_t directly from x_0 using the closed-form DDPM equation, without
    clamping to the displayable range.

    Kept separate from forward_noise() because the clamp is a *display*
    concession, not part of the process: x_t is genuinely unbounded, and the
    pixel histograms are honest about that only if they see the raw values.
    """
    eps = rng.normal(size=x0.shape)
    return np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * eps


def forward_noise(x0, alpha_bar_t, rng):
    """Sample x_t and clamp it to the displayable [-1, 1] range."""
    return np.clip(forward_noise_raw(x0, alpha_bar_t, rng), -1.0, 1.0)


def crop_to_content(x0, diff_threshold=0.08, margin_px=10):
    """
    Crop x0 down to the minimal rectangle containing its actual content,
    trimming empty background margins (e.g. white padding around a logo
    or drawing), then pad back out by exactly margin_px pixels on each
    side (not a percentage - a fixed pixel amount, so it doesn't scale
    up into visible whitespace on high-resolution images).

    Real photos/scans rarely have a perfectly pure-white background -
    JPEG artifacts and scan noise make individual pixels dip away from
    pure white all over the image, which would defeat a naive "pixel !=
    white" check. To be robust to that, the background color is sampled
    from the image's own corners (whatever that color actually is, not
    assumed to be pure white) and a pixel only counts as content if it
    differs from that background by more than diff_threshold.
    """
    h, w = x0.shape[:2]
    cs = max(2, int(round(0.02 * min(h, w))))  # corner sample size
    corners = np.concatenate([
        x0[:cs, :cs].reshape(-1, 3),
        x0[:cs, -cs:].reshape(-1, 3),
        x0[-cs:, :cs].reshape(-1, 3),
        x0[-cs:, -cs:].reshape(-1, 3),
    ], axis=0)
    bg_color = np.median(corners, axis=0)

    diff = np.abs(x0 - bg_color).max(axis=2)
    non_bg = diff > diff_threshold
    if not non_bg.any():
        return x0  # nothing detected (e.g. a blank image) - leave untouched

    # any non-background pixel in a row/column counts - this correctly
    # catches thin content like the tapering top/bottom of a circle,
    # which a "minimum fraction of the row" check would clip off.
    rows = np.where(non_bg.any(axis=1))[0]
    cols = np.where(non_bg.any(axis=0))[0]

    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()

    r0 = max(0, r0 - margin_px)
    r1 = min(h - 1, r1 + margin_px)
    c0 = max(0, c0 - margin_px)
    c1 = min(w - 1, c1 + margin_px)

    return x0[r0:r1 + 1, c0:c1 + 1]


def tighten_saved_png(path, margin_px=OUTPUT_PADDING_PX):
    """
    Post-process a saved matplotlib figure: crop the whole PNG down to
    its actual rendered content plus exactly margin_px pixels on every
    side. matplotlib's bbox_inches="tight" + pad_inches doesn't give
    reliable pixel-exact control, so this does it directly on the saved
    file instead.
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    non_white = np.any(arr < 250, axis=2)
    if not non_white.any():
        return  # blank image, nothing to do

    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    h, w = arr.shape[:2]

    r0 = max(0, rows.min() - margin_px)
    r1 = min(h - 1, rows.max() + margin_px)
    c0 = max(0, cols.min() - margin_px)
    c1 = min(w - 1, cols.max() + margin_px)

    img.crop((c0, r0, c1 + 1, r1 + 1)).save(path)


# ---------------------------------------------------------------------------
# Schematic forward/backward diagram (boxes + arrows, like the sketch)
# ---------------------------------------------------------------------------
def make_diagram(x0, alpha_bars, timesteps, rng, out_path):
    """
    timesteps must have exactly 4 entries: [0, t-1, t, T-1], i.e.
    the original image, the two adjacent steps the q/p_theta labels
    refer to, and the fully noised image.
    """
    import matplotlib.patches as mpatches

    assert len(timesteps) == 4, "expects exactly [0, t-1, t, T-1]"
    labels = [rf"${VEC_X}_0$", rf"${VEC_X}_{{t-1}}$", rf"${VEC_X}_t$", rf"${VEC_X}_T$"]

    # Sized for projection: this figure is shown on a slide, where the
    # limiting factor is the back row of the room, not page width. One size
    # for every label - no visual hierarchy between them, so the diagram
    # reads as a single uniform piece of notation.
    LABEL_FS = 26
    TRANSITION_FS = LABEL_FS   # q(...) and p_theta(...)
    STATE_FS = LABEL_FS        # x_0, x_{t-1}, x_t, x_T under the boxes
    DOTS_FS = LABEL_FS         # the "..." over the skipped-step arrows
    DIRECTION_FS = LABEL_FS    # "Forward" / "Backward"

    # Preserve the source image's aspect ratio so boxes don't stretch/skew it.
    img_h, img_w = x0.shape[0], x0.shape[1]
    aspect = img_w / img_h
    bh = 2.4                       # box height (data units)
    bw = bh * aspect                # box width, matched to image aspect ratio

    gap = max(1.4, 0.7 * bw)        # horizontal gap between boxes, scales with box width
    x_positions = [i * (bw + gap) for i in range(4)]
    y_bottom = 2.1

    fig_width = 4 * bw + 3 * gap + 1.6
    fig_height = y_bottom + bh + 4.2   # extra headroom for the enlarged labels
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-0.8, x_positions[-1] + bw + 0.8)
    ax.set_ylim(-1.2, fig_height - 0.6)   # room below for the q(...) label
    ax.set_aspect("equal", adjustable="box")   # guarantees no internal image skew
    ax.axis("off")

    for i, (t, x) in enumerate(zip(timesteps, x_positions)):
        img_t = x0 if t == 0 else forward_noise(x0, alpha_bars[t], rng)
        ax.imshow(to_display(img_t), extent=(x, x + bw, y_bottom, y_bottom + bh), zorder=2)
        ax.add_patch(mpatches.Rectangle((x, y_bottom), bw, bh, fill=False,
                                         edgecolor="black", linewidth=1.6, zorder=3))
        ax.text(x + bw / 2, y_bottom - 0.35, labels[i], ha="center", va="top",
                fontsize=STATE_FS)

    mid_y = y_bottom + bh / 2
    box1_c = x_positions[1] + bw / 2   # center of x_{t-1}
    box2_c = x_positions[2] + bw / 2   # center of x_t

    # skipped-step gaps (x_0 -> ... -> x_{t-1}, and x_t -> ... -> x_T): dotted arrow + "..."
    for i in (0, 2):
        x_a, x_b = x_positions[i] + bw, x_positions[i + 1]
        ax.annotate("", xy=(x_b, mid_y), xytext=(x_a, mid_y),
                    arrowprops=dict(arrowstyle="->", linestyle=(0, (1, 2)),
                                     color="black", lw=1.4))
        # va="bottom" anchors the glyphs above the arrow line instead of
        # letting the baseline sit on it, so the dots never touch the dashes.
        ax.text((x_a + x_b) / 2, mid_y + 0.18, "...", ha="center", va="bottom",
                fontsize=DOTS_FS)

    # adjacent single-step gap (x_{t-1} -> x_t): plain solid arrow
    x_a, x_b = x_positions[1] + bw, x_positions[2]
    ax.annotate("", xy=(x_b, mid_y), xytext=(x_a, mid_y),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4))

    # labeled curved arrows anchored specifically between x_{t-1} and x_t (single step)
    top_y = y_bottom + bh + 0.25
    bot_y = y_bottom - 0.9

    # Both labels sit clear of their arc rather than at a guessed offset: an
    # "arc3" connection is a quadratic Bezier whose control point is offset
    # perpendicular by rad * |AB|, so the curve itself peaks at half of that.
    # Placing each label past the peak keeps text and arrow from overlapping
    # no matter how the box geometry or font size changes.
    ARC_RAD = 0.5
    LABEL_CLEARANCE = 0.35
    arc_peak = ARC_RAD * abs(box2_c - box1_c) / 2
    label_offset = arc_peak + LABEL_CLEARANCE

    backward_arrow = mpatches.FancyArrowPatch(
        (box2_c, top_y), (box1_c, top_y),
        connectionstyle=f"arc3,rad={ARC_RAD}", arrowstyle="-|>", mutation_scale=20,
        color="black", lw=1.5, zorder=1)
    ax.add_patch(backward_arrow)
    ax.text((box1_c + box2_c) / 2, top_y + label_offset,
            rf"$p_\theta({VEC_X}_{{t-1}} \mid {VEC_X}_t)$",
            ha="center", va="bottom", fontsize=TRANSITION_FS)

    forward_arrow = mpatches.FancyArrowPatch(
        (box1_c, bot_y), (box2_c, bot_y),
        connectionstyle=f"arc3,rad={ARC_RAD}", arrowstyle="-|>", mutation_scale=20,
        color="black", lw=1.5, zorder=1)
    ax.add_patch(forward_arrow)
    ax.text((box1_c + box2_c) / 2, bot_y - label_offset,
            rf"$q({VEC_X}_t \mid {VEC_X}_{{t-1}})$",
            ha="center", va="top", fontsize=TRANSITION_FS)

    # These two are plain prose, not notation, so they keep matplotlib's own
    # sans-serif italic instead of being pulled into the LaTeX/serif setup that
    # the math labels use (usetex=False overrides the global rcParam per-artist).
    label_kw = dict(ha="left", fontsize=DIRECTION_FS, style="italic",
                    family="sans-serif", usetex=False)
    backward_text = ax.text(-0.5, top_y + 0.35, "Reverse", **label_kw)
    forward_text = ax.text(-0.5, bot_y - 0.6, "Forward", **label_kw)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def text_right_edge(txt_obj):
        bbox = txt_obj.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        return bbox.x1, (bbox.y0 + bbox.y1) / 2

    bx, by = text_right_edge(backward_text)
    fx, fy = text_right_edge(forward_text)
    bx -= 0.22  # italic bbox overshoots the visible glyph ink
    fx -= 0.22

    arrow_len = 0.8   # scaled up alongside the enlarged "Forward"/"Backward" text

    # simple straight arrow to the RIGHT of "Backward", pointing LEFT to match
    # the direction of the backward process (x_t -> x_{t-1})
    backward_label_arrow = mpatches.FancyArrowPatch(
        (bx + 0.1 + arrow_len, by), (bx + 0.1, by),
        arrowstyle="-|>", mutation_scale=18, color="black", lw=1.8, zorder=1)
    ax.add_patch(backward_label_arrow)

    # simple straight arrow to the RIGHT of "Forward"
    forward_label_arrow = mpatches.FancyArrowPatch(
        (fx + 0.1, fy), (fx + 0.1 + arrow_len, fy),
        arrowstyle="-|>", mutation_scale=18, color="black", lw=1.8, zorder=1)
    ax.add_patch(forward_label_arrow)

    ax.set_xlim(-0.8, x_positions[-1] + bw + 0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    tighten_saved_png(out_path)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Closed-form equation diagram:
#   sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * epsilon  =  x_t
# shown as actual images with the scalar coefficients and operators between
# them, so the "recipe" for the closed-form sampling equation is visible.
# ---------------------------------------------------------------------------
def make_equation_diagram(x0, alpha_bars, t, rng, out_path):
    import matplotlib.patches as mpatches

    a_bar = alpha_bars[t]
    eps = rng.normal(size=x0.shape)
    # eps is not an image and has no data range of its own; mapping it through
    # eps/6 + 0.5 is purely a device for making the static viewable, so it goes
    # straight to display space rather than through to_display().
    noise_vis = np.clip(eps / 6.0 + 0.5, 0.0, 1.0)
    xt = np.sqrt(a_bar) * x0 + np.sqrt(1.0 - a_bar) * eps

    img_h, img_w = x0.shape[0], x0.shape[1]
    aspect = img_w / img_h
    bh = 2.6
    bw = bh * aspect
    y_bottom = 1.1

    fig, ax = plt.subplots(figsize=(18, bh + 2.6))
    ax.axis("off")

    # sequence of (kind, content/image, label_or_None, slot_width, fontsize)
    sym_fs, coef_fs, label_fs = 24, 20, 15
    slots = [
        ("text", r"$\sqrt{\bar\alpha_t}$", None, 1.3, coef_fs),
        ("text", r"$\times$", None, 0.55, sym_fs),
        ("box", to_display(x0), rf"${VEC_X}_0$", bw, label_fs),
        ("text", r"$+$", None, 0.65, sym_fs),
        ("text", r"$\sqrt{1-\bar\alpha_t}$", None, 1.7, coef_fs),
        ("text", r"$\times$", None, 0.55, sym_fs),
        ("box", noise_vis, rf"${VEC_EPS} \sim \mathcal{{N}}({VEC_ZERO}, {VEC_I})$", bw, label_fs),
        ("text", r"$=$", None, 0.65, sym_fs),
        ("box", to_display(xt), rf"${VEC_X}_t$", bw, label_fs),
    ]

    margin = 0.3
    cursor = 0.0
    mid_y = y_bottom + bh / 2

    for kind, content, label, width, fs in slots:
        if kind == "text":
            ax.text(cursor + width / 2, mid_y, content, ha="center", va="center", fontsize=fs)
        else:  # box
            ax.imshow(content, extent=(cursor, cursor + width, y_bottom, y_bottom + bh), zorder=2)
            ax.add_patch(mpatches.Rectangle((cursor, y_bottom), width, bh, fill=False,
                                             edgecolor="black", linewidth=1.6, zorder=3))
            ax.text(cursor + width / 2, y_bottom - 0.3, label, ha="center", va="top", fontsize=fs)
        cursor += width + margin

    ax.set_xlim(-margin, cursor)
    ax.set_ylim(0, y_bottom + bh + 1.2)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    tighten_saved_png(out_path)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Noising-steps figure, with or without the pixel-density row
# ---------------------------------------------------------------------------
def render_noising_steps(t_indices, alpha_bars, shown, hist_values, hist_range,
                         panel_aspect, out_path, with_histograms):
    """
    Render the forward-process snapshots as one figure.

    with_histograms adds a second row showing the distribution of the pixel
    values behind each image. The images themselves are passed in rather than
    sampled here, so the two variants are guaranteed to show the same draws.
    """
    n = len(t_indices)
    n_rows = 2 if with_histograms else 1
    fig, axes = plt.subplots(
        n_rows, n,
        figsize=(3 * n, 3 * panel_aspect * n_rows + 1.4),
        squeeze=False)

    for col, t in enumerate(t_indices):
        ax_img = axes[0, col]
        ax_img.imshow(shown[col])
        ax_img.set_title(rf"$t = {t}$" + "\n" + rf"$\bar{{\alpha}}_t = {alpha_bars[t]:.3f}$",
                         fontsize=HIST_TITLE_FS)
        ax_img.axis("off")

        if not with_histograms:
            continue

        ax_hist = axes[1, col]
        ax_hist.hist(hist_values[col].ravel(), bins=HIST_BINS, range=hist_range,
                     density=True, color="0.30", edgecolor="none")
        ax_hist.set_box_aspect(panel_aspect)
        ax_hist.set_xlim(*hist_range)
        ax_hist.tick_params(axis="x", labelsize=HIST_TICK_FS)
        # Each panel is autoscaled to its own peak: the t=0 distribution is two
        # near-delta spikes (black and white pixels) that are orders of
        # magnitude taller than the later bell shapes, so a shared y-axis would
        # flatten every other panel into a invisible line. The y-axis therefore
        # carries no numbers - these panels compare shape, not height.
        ax_hist.set_yticks([])
        for side in ("top", "right", "left"):
            ax_hist.spines[side].set_visible(False)
        if col == 0:
            # Axis names on the first panel only - they hold for the whole
            # row, and repeating them five times is noise on a slide.
            ax_hist.set_ylabel("pixel density", fontsize=HIST_LABEL_FS)
            ax_hist.set_xlabel("pixel value", fontsize=HIST_LABEL_FS)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    tighten_saved_png(out_path)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    img = Image.open(IMAGE_PATH).convert("RGB")
    x0 = np.asarray(img, dtype=np.float64) / 255.0
    x0 = crop_to_content(x0)   # runs on [0, 1]: its thresholds are tuned there
    x0 = to_model(x0)          # from here on, everything is in [-1, 1]

    alpha_bars = make_schedule(T, BETA_START, BETA_END)

    # timesteps to show, evenly spaced across the schedule, including t=0
    t_indices = np.linspace(0, T - 1, N_STEPS_SHOWN).astype(int)

    # Draw every sample first: the histogram row needs the raw (unclamped)
    # values while the image row needs the clamped ones, and both must come
    # from the *same* draw - re-sampling per row would show a histogram that
    # doesn't belong to the image above it.
    raw = [x0 if t == 0 else forward_noise_raw(x0, alpha_bars[t], rng)
           for t in t_indices]
    clipped = [np.clip(r, -1.0, 1.0) for r in raw]
    shown = [to_display(c) for c in clipped]
    # Both options stay in model space, so the histogram axis is on the same
    # [-1, 1] scale the paper states - never display space.
    hist_values = clipped if HIST_USE_CLIPPED else raw

    # A common x-range across all five panels - otherwise each histogram is
    # drawn on its own axis and the spreading of the distribution, which is
    # the whole point of the row, becomes invisible.
    #
    # Taken from percentiles rather than min/max: with ~2M samples per panel
    # the Gaussian tail reaches past 4.5 sigma, and letting those few pixels
    # set the range squeezes every actual distribution into the middle third
    # of its panel. The 0.1/99.9 cut keeps the visible shape large.
    lo = min(float(np.percentile(v, 0.1)) for v in hist_values)
    hi = max(float(np.percentile(v, 99.9)) for v in hist_values)
    pad = 0.05 * (hi - lo)
    hist_range = (lo - pad, hi + pad)

    img_h, img_w = x0.shape[0], x0.shape[1]
    panel_aspect = img_h / img_w   # so the histograms match the images exactly

    # Both variants are rendered from the *same* samples above, so the plain
    # figure and the histogram figure show pixel-for-pixel identical images.
    render_noising_steps(t_indices, alpha_bars, shown, hist_values, hist_range,
                         panel_aspect, OUTPUT_PATH, with_histograms=True)
    render_noising_steps(t_indices, alpha_bars, shown, hist_values, hist_range,
                         panel_aspect, OUTPUT_PATH_PLAIN, with_histograms=False)

    make_diagram(x0, alpha_bars, DIAGRAM_TIMESTEPS, rng, DIAGRAM_OUTPUT_PATH)
    make_equation_diagram(x0, alpha_bars, EQUATION_TIMESTEP, rng, EQUATION_OUTPUT_PATH)


if __name__ == "__main__":
    main()