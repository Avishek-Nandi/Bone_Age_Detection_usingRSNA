# infer_and_visualize.py
from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ultralytics import YOLO

from utils_preprocess import read_gray, pseudo_hand_mask, apply_mask_and_crop, clahe_gray

def get_mask_from_yolo(yolo: YOLO, image_path: str, fallback_img_gray: np.ndarray) -> np.ndarray:
    pred = yolo.predict(source=image_path, task="segment", verbose=False)[0]
    if pred.masks is not None and len(pred.masks.data) > 0:
        m = pred.masks.data[0].cpu().numpy()
        return (m * 255).astype(np.uint8)
    return pseudo_hand_mask(fallback_img_gray)

def predict_age(regressor: tf.keras.Model, roi_gray: np.ndarray, male: int, img_size: int) -> float:
    roi = tf.image.resize(roi_gray[..., None], (img_size, img_size)).numpy().astype(np.uint8)
    roi3 = np.repeat(roi, 3, axis=-1).astype(np.float32)
    pred = regressor.predict({"image": roi3[None, ...], "male": np.array([[male]], dtype=np.float32)}, verbose=0)
    return float(pred.squeeze())

def overlay_mask(img_gray: np.ndarray, mask_255: np.ndarray) -> np.ndarray:
    # simple overlay: brighten ROI
    out = img_gray.copy().astype(np.float32)
    out[mask_255 > 0] = np.clip(out[mask_255 > 0] * 1.2, 0, 255)
    return out.astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", type=str, required=True)
    ap.add_argument("--male", type=int, default=1, help="1=male, 0=female (used by regressor)")
    ap.add_argument("--yolo_weights", type=str, required=True)
    ap.add_argument("--regressor", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--out_dir", type=str, default="outputs/viz")
    ap.add_argument("--clahe", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img = read_gray(args.image_path)
    yolo = YOLO(args.yolo_weights)
    reg = tf.keras.models.load_model(args.regressor)

    mask = get_mask_from_yolo(yolo, args.image_path, img)
    overlay = overlay_mask(img, mask)
    roi = apply_mask_and_crop(img, mask, margin=12)

    age_plain = predict_age(reg, roi, args.male, args.img_size)

    if args.clahe:
        roi_clahe = clahe_gray(roi, clip_limit=2.0, tile_grid_size=8)
        age_clahe = predict_age(reg, roi_clahe, args.male, args.img_size)
    else:
        roi_clahe = None
        age_clahe = None

    # Plot
    ncols = 4 if args.clahe else 3
    plt.figure(figsize=(4.8 * ncols, 4.8))

    ax1 = plt.subplot(1, ncols, 1)
    ax1.imshow(img, cmap="gray")
    ax1.set_title("Original")
    ax1.axis("off")

    ax2 = plt.subplot(1, ncols, 2)
    ax2.imshow(overlay, cmap="gray")
    ax2.set_title("Segmentation Overlay")
    ax2.axis("off")

    ax3 = plt.subplot(1, ncols, 3)
    ax3.imshow(roi, cmap="gray")
    ax3.set_title(f"Cleaned ROI\nPred: {age_plain:.1f} months")
    ax3.axis("off")

    if args.clahe:
        ax4 = plt.subplot(1, ncols, 4)
        ax4.imshow(roi_clahe, cmap="gray")
        ax4.set_title(f"ROI + CLAHE\nPred: {age_clahe:.1f} months")
        ax4.axis("off")

    out_path = os.path.join(args.out_dir, "inference_viz.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved visualization:", out_path)
    print("Prediction (plain):", age_plain)
    if args.clahe:
        print("Prediction (CLAHE):", age_clahe)

if __name__ == "__main__":
    main()
