#!/usr/bin/env python3
r"""
latent_visualisation.py

Encodes a single image into the latent space of a pretrained first-stage
autoencoder from Rombach et al. (2022) and visualises the resulting latent
z = E(x), optionally together with the reconstruction D(E(x)).

The latent is not an image. Its entries are unbounded real numbers, so any
depiction requires an explicit normalisation choice; the one used here is
recorded in the printed summary and must be stated in the figure caption.

The combined figure is written at exactly the width it is meant to occupy in
the document (--width-in), so that no scaling happens in LaTeX and the label
size matches the body text.  Measure your text width once with \the\linewidth
(pt / 72.27 = inches) and pass the desired fraction of it.

Usage
-----
    python src/latent_visualisation.py --width-in 4.5
    python src/latent_visualisation.py --image logo.png --model kl-f8 --norm minmax

--image is looked up in images/, results are written to images/latent_out/.

Requires: torch, diffusers, pillow, numpy, matplotlib
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# All inputs and outputs live in images/ at the repo root, independent of cwd.
IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"


# ----------------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------------
# vq-f4 : VQ-regularised, f = 4, c = 3, |Z| = 8192.  This is the first stage
#         used for most LDM experiments in the paper (cf. Tab. 8, Tab. 12).
#         c = 3 means the latent can be shown directly as a three-channel image.
# kl-f8 : KL-regularised, f = 8, c = 4.  Four channels, so it has to be shown
#         as four separate greyscale maps.
MODELS = {
    "vq-f4": dict(repo="CompVis/ldm-celebahq-256", subfolder="vqvae", kind="vq"),
    "kl-f8": dict(repo="stabilityai/sd-vae-ft-mse", subfolder=None, kind="kl"),
}


def load_autoencoder(name, device, dtype=torch.float32):
    from diffusers import AutoencoderKL, VQModel

    cfg = MODELS[name]
    if cfg["kind"] == "vq":
        model = VQModel.from_pretrained(
            cfg["repo"], subfolder=cfg["subfolder"], torch_dtype=dtype
        )
    else:
        kwargs = dict(torch_dtype=dtype)
        if cfg["subfolder"]:
            kwargs["subfolder"] = cfg["subfolder"]
        model = AutoencoderKL.from_pretrained(cfg["repo"], **kwargs)
    return model.to(device).eval(), cfg["kind"]


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def load_image(path, resolution):
    """Load an image, pad to square on white, resize, scale to [-1, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w != h:  # pad to square so the latent grid is square as well
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        img = canvas
    img = img.resize((resolution, resolution), Image.LANCZOS)
    x = np.asarray(img, dtype=np.float32) / 255.0          # [0, 1], HWC
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1CHW
    return x * 2.0 - 1.0                                   # [-1, 1]


def to_display(t):
    """[-1, 1] tensor -> [0, 1] HWC numpy array."""
    t = (t.squeeze(0).permute(1, 2, 0).float().cpu().numpy() + 1.0) / 2.0
    return np.clip(t, 0.0, 1.0)


def psnr(a, b):
    """PSNR in dB between two [0, 1] arrays."""
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


# ----------------------------------------------------------------------------
# Latent normalisation for display
# ----------------------------------------------------------------------------
def normalise_latent(z, mode, p=0.5):
    """
    Map a latent tensor (1, c, h, w) to [0, 1] for display.

    minmax     : per-channel min-max over the whole latent.
    percentile : per-channel clipping at the p / (100 - p) percentiles first,
                 then min-max.  Less sensitive to isolated outliers.

    Both are display conventions and carry no meaning beyond making the
    latent visible.  State whichever is used in the caption.
    """
    z = z.squeeze(0).float().cpu().numpy()  # (c, h, w)
    out = np.empty_like(z)
    for i, ch in enumerate(z):
        if mode == "percentile":
            lo, hi = np.percentile(ch, [p, 100.0 - p])
            ch = np.clip(ch, lo, hi)
        else:
            lo, hi = ch.min(), ch.max()
        out[i] = (ch - lo) / (hi - lo + 1e-8)
    return out  # (c, h, w) in [0, 1]


def save_png(arr, path, upscale=1):
    """arr: (h, w) or (h, w, c) in [0, 1].  Nearest-neighbour upscaling only."""
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    img.save(path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=Path("unnoised.jpg"), type=Path,
                    help="input image, relative to images/")
    ap.add_argument("--outdir", default=Path("latent_out"), type=Path,
                    help="output directory, relative to images/")
    ap.add_argument("--model", default="vq-f4", choices=list(MODELS))
    ap.add_argument("--resolution", default=256, type=int)
    ap.add_argument("--norm", default="percentile", choices=["minmax", "percentile"])
    ap.add_argument("--percentile", default=0.5, type=float)
    ap.add_argument("--upscale", default=4, type=int,
                    help="nearest-neighbour factor for the standalone latent PNG")
    ap.add_argument("--width-in", default=4.5, type=float,
                    help="figure width in inches; use this same value in "
                         "\\includegraphics[width=...in] so nothing gets rescaled")
    ap.add_argument("--fontsize", default=8.0, type=float,
                    help="panel label size in pt, matched to the document")
    ap.add_argument("--no-reconstruction", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.image = IMAGES_DIR / args.image
    args.outdir = IMAGES_DIR / args.outdir

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    args.outdir.mkdir(parents=True, exist_ok=True)

    model, kind = load_autoencoder(args.model, device)

    # --- report the actual configuration instead of trusting the model name ---
    c_cfg = getattr(model.config, "latent_channels", None)
    n_down = len(model.config.block_out_channels) - 1
    f_cfg = 2 ** n_down
    print(f"model           : {MODELS[args.model]['repo']} ({kind}-regularised)")
    print(f"downsampling f  : {f_cfg}")
    print(f"latent channels : {c_cfg}")

    x = load_image(args.image, args.resolution).to(device)

    with torch.no_grad():
        if kind == "vq":
            # Rombach et al., App. F: for a VQ-regularised space, z is taken
            # *before* the quantisation layer, which is absorbed into D.
            z = model.encode(x).latents
            rec = None if args.no_reconstruction else model.decode(z).sample
        else:
            # KL-regularised: z = E_mu + E_sigma * eps.  The posterior mean is
            # used here so that the figure is deterministic; say so in the caption.
            posterior = model.encode(x).latent_dist
            z = posterior.mean
            rec = None if args.no_reconstruction else model.decode(z).sample

    b, c, h, w = z.shape
    f_eff = args.resolution // h
    print(f"\nx shape         : {tuple(x.shape[1:])}")
    print(f"z shape         : {(c, h, w)}   (effective f = {f_eff})")
    print(f"compression     : {np.prod(x.shape[1:]) / (c * h * w):.2f}x fewer entries")
    print(f"z overall       : mean {z.mean():+.4f}  std {z.std():.4f}  "
          f"min {z.min():+.4f}  max {z.max():+.4f}")
    for i in range(c):
        zi = z[0, i]
        print(f"  channel {i}     : mean {zi.mean():+.4f}  std {zi.std():.4f}  "
              f"min {zi.min():+.4f}  max {zi.max():+.4f}")

    # ------------------------------------------------------------------ save
    x_disp = to_display(x)
    z_disp = normalise_latent(z, args.norm, args.percentile)  # (c, h, w)

    np.save(args.outdir / "latent_raw.npy", z.cpu().numpy())

    if c == 3:
        save_png(np.transpose(z_disp, (1, 2, 0)),
                 args.outdir / "latent_rgb.png", args.upscale)
    for i in range(c):
        save_png(z_disp[i], args.outdir / f"latent_ch{i}.png", args.upscale)

    rec_disp, rec_psnr = None, None
    if rec is not None:
        rec_disp = to_display(rec)
        rec_psnr = psnr(x_disp, rec_disp)
        print(f"\nPSNR D(E(x)) vs x : {rec_psnr:.2f} dB   <- put this in the caption")
        save_png(rec_disp, args.outdir / "reconstruction.png")

    # ------------------------------------------------------- combined figure
    # Panel labels stay single-line; the PSNR belongs in the caption, where it
    # does not force extra vertical space at small figure widths.
    panels = [("$\\mathbf{x}$", x_disp)]
    if c == 3:
        panels.append(("$\\mathbf{z} = \\mathcal{E}(\\mathbf{x})$",
                       np.transpose(z_disp, (1, 2, 0))))
    else:
        panels += [(f"$\\mathbf{{z}}$, ch. {i}", z_disp[i]) for i in range(c)]
    if rec_disp is not None:
        panels.append(("$\\mathcal{D}(\\mathcal{E}(\\mathbf{x}))$", rec_disp))

    n = len(panels)
    panel_w = args.width_in / n
    title_h = 1.6 * args.fontsize / 72.0          # room for one line of label
    fig, axes = plt.subplots(1, n, figsize=(args.width_in, panel_w + title_h))
    axes = np.atleast_1d(axes)
    for ax, (title, arr) in zip(axes, panels):
        ax.imshow(arr, cmap=None if arr.ndim == 3 else "gray",
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, fontsize=args.fontsize, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
    # tight_layout would override the width set above
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 - title_h / (panel_w + title_h),
                        wspace=0.04)
    for ext in ("pdf", "png"):
        fig.savefig(args.outdir / f"latent_figure.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.01)
    print(f"\nfigure width    : {args.width_in} in  ->  "
          f"\\includegraphics[width={args.width_in}in]{{...}}")
    print(f"written to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()