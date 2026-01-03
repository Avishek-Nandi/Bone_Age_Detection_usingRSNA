# eval_regressor_tf.py
from __future__ import annotations
import os
import json
import argparse
import pandas as pd
import tensorflow as tf
from ultralytics import YOLO

from train_regressor_tf import make_dataset  # reuse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", type=str, required=True)
    ap.add_argument("--img_dir", type=str, required=True)
    ap.add_argument("--yolo_weights", type=str, required=True)
    ap.add_argument("--regressor", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.splits_csv)
    test_df = df[df["split"] == "test"].copy()

    yolo = YOLO(args.yolo_weights)
    test_ds = make_dataset(test_df, args.img_dir, yolo, args.img_size, args.batch, training=False)

    model = tf.keras.models.load_model(args.regressor)
    out = model.evaluate(test_ds, verbose=1, return_dict=True)

    os.makedirs("outputs/metrics", exist_ok=True)
    out_path = "outputs/metrics/regressor_test_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in out.items()}, f, indent=2)

    print("Saved:", out_path)
    print(out)

if __name__ == "__main__":
    main()
