"""
Tiny DDPM that memorizes a single image and demonstrates the reverse
(denoising) process: pure Gaussian noise -> reconstructed image.

Designed to run on YOUR machine (not a sandbox), so it:
  - uses your image at its native resolution (only pads, never resizes/crops,
    so every original pixel is preserved)
  - auto-detects CUDA / Apple Silicon (MPS) / CPU
  - uses a larger U-Net than a tiny demo needs, since real hardware can
    afford it, which helps preserve fine detail (e.g. small text) at
    higher resolution

Setup:
    pip install -r requirements.txt

Run:
    python src/denoising.py --image your_logo.png   # looked up in images/

If your image is very large (say, >512px) and training is too slow even
on GPU, you can optionally cap it without hardcoding a small size:
    python src/denoising.py --image your_logo.png --max_side 256
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler

# how many times the U-Net below downsamples (len(down_block_types) - 1)
DOWNSAMPLE_STAGES = 3
PAD_MULTIPLE = 2 ** DOWNSAMPLE_STAGES  # = 8

# Display scale for the epsilon row of the figure. eps_theta is an unbounded
# tensor, not an image, so showing it needs a convention: values of +-3 span
# the full black-to-white range, i.e. eps/6 + 0.5. The scale is FIXED across
# all panels on purpose - normalising each panel to its own range would make
# a confident prediction and a near-zero one look identical, hiding exactly
# the change across timesteps the row is there to show.
EPS_DISPLAY_SCALE = 3.0

# Images are read from and figures written to images/ at the repo root,
# independent of the working directory.
ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT_DIR / "images"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pad_to_multiple(img: Image.Image, multiple: int) -> Image.Image:
    """Pad (never crop/resize) so H and W are divisible by `multiple`,
    which the U-Net's down/up-sampling requires. Uses the image's own
    corner pixel as background so no hard artificial border is introduced."""
    w, h = img.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple
    if (new_w, new_h) == (w, h):
        return img
    bg = img.getpixel((0, 0))
    canvas = Image.new(img.mode, (new_w, new_h), bg)
    off = ((new_w - w) // 2, (new_h - h) // 2)
    canvas.paste(img, off)
    return canvas


def load_image(path, max_side=None):
    img = Image.open(path)
    mode = "L" if img.mode in ("L", "1") else "RGB"
    img = img.convert(mode)

    if max_side is not None:
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            print(f"Note: downsampled to fit --max_side {max_side} -> {img.size}")

    orig_size = img.size
    img = pad_to_multiple(img, PAD_MULTIPLE)
    if img.size != orig_size:
        print(f"Padded {orig_size} -> {img.size} (multiple of {PAD_MULTIPLE} required by the U-Net)")

    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2 - 1  # [-1, 1]
    channels = 1 if mode == "L" else 3
    if mode == "L":
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    else:
        tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return tensor, channels, img.size


def build_model(image_size_hw, channels, gradient_checkpointing=False):
    h, w = image_size_hw
    model = UNet2DModel(
        sample_size=(h, w),
        in_channels=channels,
        out_channels=channels,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        add_attention=False,  # the mid-block includes attention by default even
                               # with no Attn*Block2D above — must disable explicitly
    )
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
    return model


def train(model, scheduler, x0, T, steps, batch_size, lr, device, use_amp=False, log_every=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if use_amp and not amp_enabled:
        print("--amp requested but no CUDA device found; continuing without it")

    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = x0.repeat(batch_size, 1, 1, 1).to(device)
        noise = torch.randn_like(batch)
        t = torch.randint(0, T, (batch_size,), device=device).long()
        noisy = scheduler.add_noise(batch, noise, t)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            pred = model(noisy, t).sample
            loss = F.mse_loss(pred, noise)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        if step % log_every == 0 or step == 1:
            print(f"step {step:5d}/{steps}  loss {loss.item():.4f}  "
                  f"avg{log_every} {np.mean(losses[-log_every:]):.4f}  "
                  f"elapsed {time.time() - t0:.1f}s")
    return losses


def save_weights(model, path, image_size_hw, channels, T, steps):
    """
    Write the trained weights plus the configuration needed to rebuild the
    same U-Net. The geometry is part of the checkpoint because sample_size,
    in_channels and out_channels are baked into the model: loading these
    weights into a model built for a different image silently fails or
    produces garbage, so --load_weights checks them rather than trusting
    that the flags were repeated correctly.
    """
    torch.save({
        "state_dict": model.state_dict(),
        "image_size_hw": tuple(image_size_hw),
        "channels": channels,
        "T": T,
        "train_steps": steps,
    }, path)
    print(f"saved weights to {path}")


def load_weights(model, path, image_size_hw, channels, T):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    saved_hw = tuple(ckpt["image_size_hw"])
    if saved_hw != tuple(image_size_hw):
        raise SystemExit(
            f"checkpoint was trained on {saved_hw[1]}x{saved_hw[0]} but the current "
            f"image is {image_size_hw[1]}x{image_size_hw[0]}; pass the same --image "
            f"and --max_side that produced it")
    if ckpt["channels"] != channels:
        raise SystemExit(
            f"checkpoint has {ckpt['channels']} channel(s), current image has {channels}")
    if ckpt["T"] != T:
        raise SystemExit(
            f"checkpoint was trained with T={ckpt['T']}, current run uses T={T}")

    model.load_state_dict(ckpt["state_dict"])
    print(f"loaded weights from {path} (trained for {ckpt['train_steps']} steps)")


@torch.no_grad()
def reverse_sample(model, scheduler, image_size_hw, channels, T, capture_fracs, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    scheduler.set_timesteps(T)
    h, w = image_size_hw
    x = torch.randn(1, channels, h, w, device=device)

    capture_at_t = sorted({int(round((1 - f) * (T - 1))) for f in capture_fracs})
    captured = {}
    eps_preds = {}

    for t in scheduler.timesteps:
        ti = t.item()
        if ti in capture_at_t:
            captured[ti] = x.clone().cpu()
        t_batch = torch.tensor([ti], device=device)
        noise_pred = model(x, t_batch).sample
        if ti in capture_at_t:
            # Captured after the forward pass but before the update, so this
            # is exactly the prediction the network made on the state stored
            # above it - the two panels of a column belong to each other.
            eps_preds[ti] = noise_pred.clone().cpu()
        x = scheduler.step(noise_pred, ti, x).prev_sample

    # The t=0 image is the finished sample, i.e. the result of the final step,
    # so it replaces the state captured at the top of that step. Its epsilon
    # panel is the prediction that produced it.
    captured[0] = x.clone().cpu()
    return captured, eps_preds, scheduler.alphas_cumprod.cpu().numpy()


def _to_hwc(arr01, channels):
    """[C,H,W] in [0,1] -> the layout imshow wants, as uint8."""
    arr01 = arr01.squeeze(0) if channels == 1 else arr01.transpose(1, 2, 0)
    return (arr01 * 255).astype(np.uint8)


def to_uint8(tensor, channels):
    """An x_t sample: [-1, 1] model range -> displayable image."""
    arr = tensor.squeeze(0).numpy()  # [C,H,W]
    return _to_hwc(np.clip((arr + 1) / 2, 0, 1), channels)


def eps_to_uint8(tensor, channels):
    """A predicted epsilon: unbounded -> displayable via EPS_DISPLAY_SCALE."""
    arr = tensor.squeeze(0).numpy()  # [C,H,W]
    arr = arr / (2 * EPS_DISPLAY_SCALE) + 0.5
    return _to_hwc(np.clip(arr, 0, 1), channels)


def _frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color("black"); spine.set_linewidth(0.8)


def make_figure(captured, eps_preds, alphas_cumprod, order, channels, out_path):
    """
    Two rows: the reverse trajectory on top, and underneath it the noise the
    network predicted at that same timestep - the quantity the model actually
    outputs, which the top row only shows the consequences of.
    """
    n = len(order)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.8), squeeze=False)
    cmap = "gray" if channels == 1 else None

    for col, t in enumerate(order):
        ax_img = axes[0, col]
        ax_img.imshow(to_uint8(captured[t], channels), cmap=cmap, vmin=0, vmax=255)
        _frame(ax_img)
        ax_img.set_title(f"t = {t}\n" + r"$\bar{\alpha}_t$" + f" = {alphas_cumprod[t]:.3f}",
                         fontsize=11)

        ax_eps = axes[1, col]
        ax_eps.imshow(eps_to_uint8(eps_preds[t], channels), cmap=cmap, vmin=0, vmax=255)
        _frame(ax_eps)

    # Row names on the left-hand panels; set_ylabel still works with the ticks
    # off, since the panels keep their frame rather than being axis("off").
    axes[0, 0].set_ylabel(r"$\mathbf{x}_t$", fontsize=13)
    axes[1, 0].set_ylabel(r"$\epsilon_\theta(\mathbf{x}_t, t)$", fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved figure to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="unnoised.jpg",
                         help="target image, relative to images/ (native resolution used)")
    parser.add_argument("--max_side", type=int, default=None,
                         help="optional cap on the longer side if your image is very large; omit to use full resolution")
    parser.add_argument("--T", type=int, default=1000, help="num diffusion timesteps")
    parser.add_argument("--steps", type=int, default=3000, help="training steps")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=3, help="sampling seed")
    parser.add_argument("--out", default="figure_reverse_process.png",
                         help="output figure, relative to images/")
    parser.add_argument("--save_weights", default="ddpm_weights.pt",
                         help="where to write the trained weights, relative to the repo root; "
                              "pass an empty string to skip saving")
    parser.add_argument("--load_weights", default=None,
                         help="load weights from this checkpoint and skip training, "
                              "e.g. to re-render the figure without training again")
    parser.add_argument("--amp", action="store_true",
                         help="use fp16 mixed precision (CUDA only) — roughly halves memory use")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                         help="trade compute for memory by not storing all activations for backward — "
                              "use this if you hit CUDA out of memory even with a small --batch_size")
    args = parser.parse_args()
    args.image = IMAGES_DIR / args.image
    args.out = IMAGES_DIR / args.out
    if args.save_weights:
        args.save_weights = ROOT_DIR / args.save_weights
    if args.load_weights:
        args.load_weights = ROOT_DIR / args.load_weights

    device = get_device()
    print(f"using device: {device}")
    if device.type == "cpu":
        import os
        torch.set_num_threads(os.cpu_count() or 1)
        print(f"CPU threads: {torch.get_num_threads()} "
              f"(no GPU detected — this will be considerably slower)")

    x0, channels, size_hw_wh = load_image(args.image, args.max_side)
    w, h = size_hw_wh
    image_size_hw = (h, w)
    x0 = x0.to(device)

    model = build_model(image_size_hw, channels, gradient_checkpointing=args.gradient_checkpointing).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=args.T, beta_schedule="linear")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"image size: {w}x{h}, channels: {channels}")
    print(f"model params: {n_params/1e6:.2f}M")

    if args.load_weights:
        load_weights(model, args.load_weights, image_size_hw, channels, args.T)
    else:
        train(model, scheduler, x0, args.T, args.steps, args.batch_size, args.lr,
              device, use_amp=args.amp)
        if args.save_weights:
            # Saved before sampling: sampling walks the full chain and is the
            # part most likely to run into the wall clock, and losing hours of
            # training to that would be avoidable.
            save_weights(model, args.save_weights, image_size_hw, channels,
                         args.T, args.steps)

    captured, eps_preds, alphas_cumprod = reverse_sample(
        model, scheduler, image_size_hw, channels, args.T,
        capture_fracs=[0.0, 0.25, 0.5, 0.75, 1.0],
        device=device, seed=args.seed,
    )
    order = sorted(captured.keys(), reverse=True)
    make_figure(captured, eps_preds, alphas_cumprod, order, channels, args.out)


if __name__ == "__main__":
    main()