# train_yolo_seg.py
from __future__ import annotations
import os
import argparse
import pandas as pd
from ultralytics import YOLO

def compute_f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_yaml", type=str, required=True)
    ap.add_argument("--model", type=str, default="yolov8n-seg.pt")  # transfer learning
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=0.0005)
    ap.add_argument("--project", type=str, default="runs")
    ap.add_argument("--name", type=str, default="rsna_hand_seg")
    args = ap.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        project=args.project,
        name=args.name,
        task="segment",
        pretrained=True,
        patience=15
    )

    # Ultralytics writes a results.csv under runs/segment/<name>/results.csv
    run_dir = results.save_dir  # pathlib
    results_csv = os.path.join(str(run_dir), "results.csv")
    if not os.path.exists(results_csv):
        print(f"Training finished, but did not find: {results_csv}")
        return

    df = pd.read_csv(results_csv)

    # Try to pick segmentation (mask) precision/recall if present, else bbox
    prec_col = next((c for c in df.columns if "metrics/precision(M)" in c), None) or \
               next((c for c in df.columns if "metrics/precision(B)" in c), None)
    rec_col = next((c for c in df.columns if "metrics/recall(M)" in c), None) or \
              next((c for c in df.columns if "metrics/recall(B)" in c), None)

    if prec_col and rec_col:
        df["F1"] = [compute_f1(p, r) for p, r in zip(df[prec_col].fillna(0), df[rec_col].fillna(0))]
    else:
        df["F1"] = 0.0

    os.makedirs("outputs/metrics", exist_ok=True)
    out_metrics = "outputs/metrics/yolo_epoch_metrics.csv"
    df.to_csv(out_metrics, index=False)

    best_row = df.iloc[df["F1"].idxmax()] if len(df) else None
    print("\nSaved epoch metrics:", out_metrics)
    if best_row is not None:
        print("\nBest epoch by F1:")
        print(best_row[["epoch", prec_col, rec_col, "F1"]].to_string())

    # Copy best.pt into models/yolo/
    best_pt = os.path.join(str(run_dir), "weights", "best.pt")
    os.makedirs("models/yolo", exist_ok=True)
    if os.path.exists(best_pt):
        dst = "models/yolo/best.pt"
        import shutil
        shutil.copy2(best_pt, dst)
        print("\nSaved best YOLO weights:", dst)
    else:
        print("\nWarning: best.pt not found at:", best_pt)

if __name__ == "__main__":
    main()
