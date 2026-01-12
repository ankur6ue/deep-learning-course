# gen_ood_digits.py
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
import torch
from pathlib import Path


OUT_DIR = Path("../../Module 2/data/ood_mnist_digits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIGITS = ["10", "11", "12", "13"]
N_PER_CLASS = 200

def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """
    Try to load a TTF font; fall back to default if not available.
    You can point this to any local .ttf you like.
    """
    try:
        # DejaVuSansMono is common on many Linux systems; change as needed.
        return ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def generate_ood_digit_image(
    text: str,
    img_size: int = 28,
    font_size_range=(18, 26),          # much bigger than before
    rotation_range=(-15, 15),          # random rotation in degrees
    scale_range=(0.9, 1.2),            # random scaling
    shear_range=(-10, 10),             # random shear in degrees
    add_noise: bool = True,
    noise_sigma_range=(0.05, 0.25),    # relative to [0, 1] pixel intensity
) -> Image.Image:
    """
    Generate an OOD 'digit' image (e.g., '10', '11', '12') that:
      - Fills most of the 28x28 patch.
      - Is randomly rotated / scaled / sheared.
      - Optionally has random Gaussian noise.

    Returns a 28x28 single-channel (grayscale) PIL Image in 'L' mode.
    """

    # --- 1. Start with a slightly larger canvas to allow for affine transforms ---
    canvas_size = img_size
    img = Image.new("L", (canvas_size, canvas_size), color=0)  # black background
    draw = ImageDraw.Draw(img)

    # --- 2. Random font size & font ---
    font_size = random.randint(*font_size_range)
    font = _load_font(font_size)

    # --- 3. Measure text size using textbbox (works on Pillow >= 10) ---
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # --- 4. Random small jitter around the center so digits don't always sit exactly centered ---
    center_x = canvas_size / 2
    center_y = canvas_size / 2
    jitter_x = random.randint(-2, 2)
    jitter_y = random.randint(-2, 2)

    x = center_x - text_w / 2 + jitter_x
    y = center_y - text_h / 2 + jitter_y

    # Draw bright text (white-ish) on black background
    draw.text((x, y), text, fill=255, font=font)

    # --- 5. Random affine transform (rotation, scale, shear) ---
    # Convert to tensor for torchvision affine
    img_tensor = F.to_tensor(img)  # shape (1, H, W), values in [0, 1]

    angle = random.uniform(*rotation_range)
    scale = random.uniform(*scale_range)
    shear = random.uniform(*shear_range)

    # center of rotation/affine
    center = (canvas_size / 2.0, canvas_size / 2.0)

    img_tensor = F.affine(
        img_tensor,
        angle=angle,
        translate=[0, 0],   # we already jittered in pixel space
        scale=scale,
        shear=[shear, 0.0],
        interpolation=F.InterpolationMode.BILINEAR,
        center=center,
        fill=0.0,           # keep background black
    )

    # --- 6. Optional additive Gaussian noise ---
    if add_noise:
        sigma = random.uniform(*noise_sigma_range)
        noise = torch.randn_like(img_tensor) * sigma
        img_tensor = img_tensor + noise
        img_tensor = img_tensor.clamp(0.0, 1.0)

    # --- 7. Convert back to 8-bit grayscale PIL Image ---
    img_np = (img_tensor.squeeze(0).numpy() * 255.0).astype(np.uint8)
    out_img = Image.fromarray(img_np, mode="L")

    # Ensure final size is exactly img_size x img_size
    if out_img.size != (img_size, img_size):
        out_img = out_img.resize((img_size, img_size), Image.BILINEAR)

    return out_img

def main():
    for d in DIGITS:
        class_dir = OUT_DIR / d
        class_dir.mkdir(exist_ok=True)
        for i in range(N_PER_CLASS):
            ood_img = generate_ood_digit_image(d)
            ood_img.save(class_dir / f"{d}_{i:04d}.png")
    print(f"Saved OOD images to {OUT_DIR}")

if __name__ == "__main__":
    main()




