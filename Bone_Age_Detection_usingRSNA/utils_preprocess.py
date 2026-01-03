# utils_preprocess.py
from __future__ import annotations
import os
import cv2
import numpy as np

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def clahe_gray(img_gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(img_gray)

def pseudo_hand_mask(img_gray: np.ndarray) -> np.ndarray:
    """
    Heuristic segmentation for hand X-ray ROI:
    - Otsu threshold
    - morphological cleanup
    - keep largest connected component
    Returns: binary mask (0/255)
    """
    # Smooth
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Otsu: decide inversion based on mean intensity
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If background becomes white, invert
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)

    # Morph close then open
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)

    # Largest CC
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(img_gray)

    # skip label 0 (background)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.zeros_like(img_gray)
    mask[labels == largest] = 255
    return mask

def mask_to_polygon_yolo(mask_255: np.ndarray, epsilon_frac: float = 0.002, max_points: int = 200) -> list[float]:
    """
    Convert a binary mask (0/255) to a YOLO polygon list: [x1,y1,x2,y2,...] normalized.
    """
    h, w = mask_255.shape[:2]
    cnts, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return []

    eps = epsilon_frac * (w + h)
    approx = cv2.approxPolyDP(cnt, eps, True)  # (N,1,2)
    pts = approx.reshape(-1, 2)

    # If too many points, downsample
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]

    poly = []
    for x, y in pts:
        poly.append(float(x) / float(w))
        poly.append(float(y) / float(h))
    return poly

def bbox_from_mask(mask_255: np.ndarray) -> tuple[int,int,int,int] | None:
    ys, xs = np.where(mask_255 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2

def apply_mask_and_crop(img_gray: np.ndarray, mask_255: np.ndarray, margin: int = 10) -> np.ndarray:
    """
    Remove background + crop tight around ROI.
    """
    bb = bbox_from_mask(mask_255)
    if bb is None:
        return img_gray.copy()

    x1, y1, x2, y2 = bb
    h, w = img_gray.shape[:2]
    x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
    x2 = min(w - 1, x2 + margin); y2 = min(h - 1, y2 + margin)

    masked = cv2.bitwise_and(img_gray, img_gray, mask=mask_255)
    cropped = masked[y1:y2+1, x1:x2+1]
    return cropped
