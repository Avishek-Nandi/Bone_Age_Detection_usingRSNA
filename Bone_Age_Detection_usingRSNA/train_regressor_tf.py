# train_regressor_tf.py
from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

from ultralytics import YOLO
from utils_preprocess import read_gray, pseudo_hand_mask, apply_mask_and_crop

def make_dataset(df: pd.DataFrame, img_root: str, yolo: YOLO, img_size: int, batch: int, training: bool):
    paths = []
    ages = []
    males = []

    for r in df.itertuples(index=False):
        p = os.path.join(img_root, r.split, f"{r.id}.png")
        if os.path.exists(p):
            paths.append(p)
            ages.append(float(r.boneage))
            males.append(float(r.male))

    ages = np.array(ages, dtype=np.float32)
    males = np.array(males, dtype=np.float32)

    def _py_load(path_t, male_t, age_t):
        path = path_t.numpy().decode("utf-8")
        img = read_gray(path)

        # Use YOLO if available; fallback to heuristic mask
        try:
            pred = yolo.predict(source=path, task="segment", verbose=False)[0]
            mask = None
            if pred.masks is not None and len(pred.masks.data) > 0:
                m = pred.masks.data[0].cpu().numpy()
                mask = (m * 255).astype(np.uint8)
            if mask is None:
                mask = pseudo_hand_mask(img)
        except Exception:
            mask = pseudo_hand_mask(img)

        roi = apply_mask_and_crop(img, mask, margin=12)
        roi = tf.image.resize(roi[..., None], (img_size, img_size)).numpy().astype(np.uint8)

        # 3-channel for EfficientNet
        roi3 = np.repeat(roi, 3, axis=-1).astype(np.float32)
        return roi3, np.array([male_t.numpy()], dtype=np.float32), np.array([age_t.numpy()], dtype=np.float32)

    def _tf_map(path, male, age):
        roi3, male_out, age_out = tf.py_function(
            _py_load, inp=[path, male, age], Tout=[tf.float32, tf.float32, tf.float32]
        )
        roi3.set_shape((img_size, img_size, 3))
        male_out.set_shape((1,))
        age_out.set_shape((1,))
        return {"image": roi3, "male": male_out}, age_out

    ds = tf.data.Dataset.from_tensor_slices((paths, males, ages))
    if training:
        ds = ds.shuffle(1024, reshuffle_each_iteration=True)
    ds = ds.map(_tf_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(img_size: int):
    img_in = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    male_in = tf.keras.Input(shape=(1,), name="male")

    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=img_in)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Concatenate()([x, male_in])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs={"image": img_in, "male": male_in}, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mae",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                 tf.keras.metrics.MeanSquaredError(name="mse")]
    )
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_csv", type=str, required=True)
    ap.add_argument("--img_dir", type=str, required=True)  # data/processed/images
    ap.add_argument("--yolo_weights", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(args.splits_csv)

    yolo = YOLO(args.yolo_weights)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    train_ds = make_dataset(train_df, args.img_dir, yolo, args.img_size, args.batch, training=True)
    val_ds = make_dataset(val_df, args.img_dir, yolo, args.img_size, args.batch, training=False)

    model = build_model(args.img_size)

    os.makedirs("models/regressor", exist_ok=True)
    ckpt_path = "models/regressor/best.keras"
    cb = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_mae", save_best_only=True, mode="min"),
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=6, restore_best_weights=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cb)

    print("Saved best regressor:", ckpt_path)
    print("Last val MAE:", history.history["val_mae"][-1])

if __name__ == "__main__":
    main()
