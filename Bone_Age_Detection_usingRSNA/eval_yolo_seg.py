# eval_yolo_seg.py
from __future__ import annotations
import os
import json
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_yaml", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    res = model.val(data=args.data_yaml, split=args.split, imgsz=args.imgsz)

    metrics = {
        "split": args.split,
        "box_map50": float(getattr(res.box, "map50", 0.0) or 0.0),
        "box_map": float(getattr(res.box, "map", 0.0) or 0.0),
        "seg_map50": float(getattr(res.seg, "map50", 0.0) or 0.0) if hasattr(res, "seg") else 0.0,
        "seg_map": float(getattr(res.seg, "map", 0.0) or 0.0) if hasattr(res, "seg") else 0.0,
    }

    os.makedirs("outputs/metrics", exist_ok=True)
    out_path = f"outputs/metrics/yolo_{args.split}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("YOLO evaluation saved:", out_path)
    print(metrics)

if __name__ == "__main__":
    main()
