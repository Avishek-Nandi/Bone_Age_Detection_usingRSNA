# prepare_dataset.py
from __future__ import annotations
import os
import glob
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from utils_preprocess import (
    ensure_dir, read_gray, pseudo_hand_mask, mask_to_polygon_yolo
)

def find_csv_with_columns(raw_dir: str) -> str:
    csvs = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
    for c in csvs:
        try:
            df = pd.read_csv(c)
        except Exception:
            continue
        cols = set(df.columns.str.lower())
        if {"id", "boneage", "male"}.issubset(cols) or {"id", "boneage", "gender"}.issubset(cols):
            return c
    raise FileNotFoundError("Could not find a CSV containing id + boneage + male/gender in data/raw")

def find_image_dir(raw_dir: str) -> str:
    # Find a folder containing many .png files
    cand = []
    for d, _, _ in os.walk(raw_dir):
        pngs = glob.glob(os.path.join(d, "*.png"))
        if len(pngs) > 1000:
            cand.append((len(pngs), d))
    if not cand:
        raise FileNotFoundError("Could not find an image directory with many .png files in data/raw")
    cand.sort(reverse=True)
    return cand[0][1]

def normalize_labels_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "gender" in df.columns and "male" not in df.columns:
        # Try map gender text
        df["male"] = df["gender"].astype(str).str.lower().map({"male": 1, "m": 1, "female": 0, "f": 0})
    df["id"] = df["id"].astype(int)
    df["boneage"] = df["boneage"].astype(float)
    df["male"] = df["male"].astype(int)
    return df[["id", "boneage", "male"]]

def make_age_bins(boneage: pd.Series, n_bins: int = 10) -> pd.Series:
    return pd.qcut(boneage, q=n_bins, labels=False, duplicates="drop")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw")
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--test_size", type=float, default=0.10)
    args = ap.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir

    csv_path = find_csv_with_columns(raw_dir)
    img_dir = find_image_dir(raw_dir)

    df = pd.read_csv(csv_path)
    df = normalize_labels_df(df)

    # Build stratification key: gender + age-bin
    df["age_bin"] = make_age_bins(df["boneage"], n_bins=12)
    df["strata"] = df["male"].astype(str) + "_" + df["age_bin"].astype(str)

    # Split train vs temp, then temp into val/test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size + args.test_size, random_state=42)
    train_idx, temp_idx = next(sss1.split(df, df["strata"]))
    df_train = df.iloc[train_idx].copy()
    df_temp = df.iloc[temp_idx].copy()

    temp_test_ratio = args.test_size / (args.val_size + args.test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=temp_test_ratio, random_state=42)
    val_idx, test_idx = next(sss2.split(df_temp, df_temp["strata"]))
    df_val = df_temp.iloc[val_idx].copy()
    df_test = df_temp.iloc[test_idx].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    splits = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Output dirs
    for sp in ["train", "val", "test"]:
        ensure_dir(os.path.join(out_dir, "images", sp))
        ensure_dir(os.path.join(out_dir, "labels", sp))

    # Copy images + create YOLO polygon labels (class=0: hand)
    missing = 0
    for row in tqdm(splits.itertuples(index=False), total=len(splits), desc="Copy + pseudo-label"):
        img_path = os.path.join(img_dir, f"{row.id}.png")
        if not os.path.exists(img_path):
            missing += 1
            continue

        dst_img = os.path.join(out_dir, "images", row.split, f"{row.id}.png")
        shutil.copy2(img_path, dst_img)

        img = read_gray(dst_img)
        mask = pseudo_hand_mask(img)
        poly = mask_to_polygon_yolo(mask)

        label_path = os.path.join(out_dir, "labels", row.split, f"{row.id}.txt")
        if poly:
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("0 " + " ".join(f"{p:.6f}" for p in poly) + "\n")
        else:
            # Empty label file (YOLO will treat as no object); keep for consistency
            open(label_path, "w", encoding="utf-8").close()

    splits_out = os.path.join(out_dir, "boneage_splits.csv")
    splits.to_csv(splits_out, index=False)

    # Write YOLO dataset YAML
    ensure_dir("configs")
    yaml_path = os.path.join("configs", "yolo_hand_seg.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n")
        f.write("  0: hand\n")

    print("\nDone.")
    print(f"- splits CSV: {splits_out}")
    print(f"- YOLO YAML  : {yaml_path}")
    if missing:
        print(f"- Missing images: {missing} (check raw dataset folder structure)")

if __name__ == "__main__":
    main()
