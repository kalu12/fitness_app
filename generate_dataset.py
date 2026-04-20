#!/usr/bin/env python3
"""
Push-up Dataset Generator
==========================
Runs YOLOv8n-pose on every GOOD/BAD annotated frame and exports a CSV
dataset ready for ML/NN training.

Each row = one frame with:
  - 17 × (x_norm, y_norm, confidence)  — raw COCO keypoints, normalized 0-1
  - Derived joint angles (elbow, body alignment, hip)
  - Derived ratios (wrist offset, head position)
  - label:     "good" | "bad"
  - label_int: 1      | 0

Usage:
  python generate_dataset.py
  python generate_dataset.py --annotations annotations.json --output dataset.csv
  python generate_dataset.py --model yolov8s-pose.pt      # larger model
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed.  Run: pip install pandas")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed.  Run: pip install ultralytics")
    sys.exit(1)


# ─── COCO-17 keypoint index lookup ───────────────────────────────────────────

KP_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
]

IDX = {name: i for i, name in enumerate(KP_NAMES)}


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _angle(a, b, c) -> float:
    """Angle (degrees) at vertex b in the path a → b → c."""
    ba  = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc  = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    n   = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n < 1e-8:
        return -1.0
    cos = np.clip(np.dot(ba, bc) / n, -1.0, 1.0)
    return float(math.degrees(math.acos(cos)))


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_features(xy_norm: np.ndarray, conf: np.ndarray) -> dict:
    """
    Given normalized keypoints (17×2) and confidences (17,), return a flat
    feature dict of raw keypoints + derived push-up features.

    Rules:
    - Any derived feature that requires a keypoint with conf < 0.3 is set to -1.0
      (acts as a "missing" sentinel — tree-based models handle this fine;
       for NNs consider imputing or masking).
    """
    features = {}

    # ── Raw keypoints ─────────────────────────────────────────────────────────
    for i, name in enumerate(KP_NAMES):
        features[f"kp_{name}_x"]    = float(xy_norm[i, 0])
        features[f"kp_{name}_y"]    = float(xy_norm[i, 1])
        features[f"kp_{name}_conf"] = float(conf[i])

    # ── Convenience accessors ─────────────────────────────────────────────────
    def pt(name):
        return xy_norm[IDX[name]]

    def c(name):
        return float(conf[IDX[name]])

    def ok(*names, threshold=0.30):
        return all(c(n) >= threshold for n in names)

    # ── Elbow angle (shoulder–elbow–wrist) ────────────────────────────────────
    # Key indicator of up vs. down position; ~160° = arms extended, ~90° = bottom
    for side in ("left", "right"):
        sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
        features[f"angle_elbow_{side}"] = (
            _angle(pt(sh), pt(el), pt(wr)) if ok(sh, el, wr) else -1.0
        )

    # ── Body alignment (shoulder–hip–ankle) ───────────────────────────────────
    # Should be ~180° for a rigid plank; sagging hips → < 180°
    for side in ("left", "right"):
        sh, hi, an = f"{side}_shoulder", f"{side}_hip", f"{side}_ankle"
        features[f"angle_body_{side}"] = (
            _angle(pt(sh), pt(hi), pt(an)) if ok(sh, hi, an) else -1.0
        )

    # ── Hip angle (shoulder–hip–knee) ─────────────────────────────────────────
    for side in ("left", "right"):
        sh, hi, kn = f"{side}_shoulder", f"{side}_hip", f"{side}_knee"
        features[f"angle_hip_{side}"] = (
            _angle(pt(sh), pt(hi), pt(kn)) if ok(sh, hi, kn) else -1.0
        )

    # ── Knee angle (hip–knee–ankle) ───────────────────────────────────────────
    for side in ("left", "right"):
        hi, kn, an = f"{side}_hip", f"{side}_knee", f"{side}_ankle"
        features[f"angle_knee_{side}"] = (
            _angle(pt(hi), pt(kn), pt(an)) if ok(hi, kn, an) else -1.0
        )

    # ── Shoulder width (normalisation denominator) ────────────────────────────
    sh_width = _dist(pt("left_shoulder"), pt("right_shoulder"))
    features["shoulder_width_norm"] = sh_width  # already in 0-1 space

    safe_w = max(sh_width, 1e-4)

    # ── Wrist horizontal offset relative to shoulder ──────────────────────────
    # Negative = wrist inside shoulder line, positive = outside; good form ≈ 0
    for side in ("left", "right"):
        sh, wr = f"{side}_shoulder", f"{side}_wrist"
        if ok(sh, wr):
            dx = float(pt(wr)[0] - pt(sh)[0])
            features[f"wrist_offset_x_{side}"] = dx / safe_w
        else:
            features[f"wrist_offset_x_{side}"] = -999.0  # missing

    # ── Wrist vertical offset relative to shoulder ────────────────────────────
    for side in ("left", "right"):
        sh, wr = f"{side}_shoulder", f"{side}_wrist"
        if ok(sh, wr):
            dy = float(pt(wr)[1] - pt(sh)[1])
            features[f"wrist_offset_y_{side}"] = dy / safe_w
        else:
            features[f"wrist_offset_y_{side}"] = -999.0

    # ── Head drop relative to shoulders ───────────────────────────────────────
    # Positive → head is BELOW shoulders (face-down start position)
    # Large negative → head excessively raised (bad neck position)
    if ok("nose", "left_shoulder", "right_shoulder"):
        shoulder_y = (pt("left_shoulder")[1] + pt("right_shoulder")[1]) / 2
        features["head_drop_norm"] = float((pt("nose")[1] - shoulder_y) / safe_w)
    else:
        features["head_drop_norm"] = -999.0

    # ── Hip height relative to shoulder–ankle line (sag / pike) ───────────────
    # For a perfect plank the hip should lie on the line shoulder→ankle.
    # Positive → hip above the line (piked), Negative → hip below (sagged)
    for side in ("left", "right"):
        sh, hi, an = f"{side}_shoulder", f"{side}_hip", f"{side}_ankle"
        if ok(sh, hi, an):
            s = np.asarray(pt(sh))
            e = np.asarray(pt(an))
            h = np.asarray(pt(hi))
            # signed perpendicular distance (normalised by body length)
            body_len = _dist(s, e) + 1e-8
            # cross-product z-component gives signed area
            cross = float((e[0] - s[0]) * (h[1] - s[1]) - (e[1] - s[1]) * (h[0] - s[0]))
            features[f"hip_deviation_{side}"] = cross / body_len
        else:
            features[f"hip_deviation_{side}"] = -999.0

    # ── Mean keypoint confidence (data quality indicator) ─────────────────────
    features["mean_conf"] = float(conf.mean())

    return features


# ─── Core pipeline ────────────────────────────────────────────────────────────

def generate(annotations_path: str, model_name: str, output_path: str,
             conf_threshold: float = 0.30):

    # Load annotations
    with open(annotations_path) as f:
        raw = json.load(f)

    labeled = {p: l for p, l in raw.items() if l in ("good", "bad")}

    if not labeled:
        print("No GOOD or BAD annotations found.")
        print(f"Run annotator.py first, then re-run this script.")
        sys.exit(1)

    n_good = sum(1 for v in labeled.values() if v == "good")
    n_bad  = sum(1 for v in labeled.values() if v == "bad")
    print(f"Annotations loaded: {n_good} GOOD  +  {n_bad} BAD  =  {len(labeled)} frames")

    # Load model
    print(f"\nLoading pose model '{model_name}'  (auto-downloads on first run) …")
    model = YOLO(model_name)
    print("Model ready.\n")

    rows    = []
    skipped = 0

    for i, (frame_path, label) in enumerate(labeled.items()):
        tag = f"[{i + 1:>4}/{len(labeled)}]"

        if not os.path.exists(frame_path):
            print(f"{tag}  MISSING   {frame_path}")
            skipped += 1
            continue

        results = model(frame_path, verbose=False, conf=conf_threshold)

        if not results or results[0].keypoints is None or len(results[0].boxes) == 0:
            print(f"{tag}  NO DETECT {Path(frame_path).name}")
            skipped += 1
            continue

        r   = results[0]
        kps = r.keypoints

        # Pick the person with the highest detection confidence
        best = int(r.boxes.conf.argmax())

        xy_px  = kps.xy[best].cpu().numpy()    # (17, 2)  pixel coords
        conf   = kps.conf[best].cpu().numpy()  # (17,)

        # Normalise by image dimensions
        img_hw = r.orig_shape  # (H, W)
        xy_norm = xy_px / np.array([img_hw[1], img_hw[0]])

        feats = extract_features(xy_norm, conf)
        feats["frame_path"] = frame_path
        feats["label"]      = label
        feats["label_int"]  = 1 if label == "good" else 0

        rows.append(feats)

        elbow_angle = (
            feats.get("angle_elbow_left", -1) + feats.get("angle_elbow_right", -1)
        ) / 2
        body_angle = (
            feats.get("angle_body_left", -1) + feats.get("angle_body_right", -1)
        ) / 2
        print(
            f"{tag}  {label.upper():<7}  "
            f"elbow≈{elbow_angle:5.1f}°  body≈{body_angle:5.1f}°  "
            f"conf={feats['mean_conf']:.2f}  {Path(frame_path).name}"
        )

    # ── Save dataset ──────────────────────────────────────────────────────────
    if not rows:
        print("\nNo rows collected — nothing to save.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Column order: meta | label | keypoints | derived features
    meta_cols = ["frame_path", "label", "label_int"]
    kp_cols   = [c for c in df.columns if c.startswith("kp_")]
    feat_cols  = [c for c in df.columns if c not in meta_cols and c not in kp_cols]
    df = df[meta_cols + kp_cols + feat_cols]

    df.to_csv(output_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_good_out = int((df.label == "good").sum())
    n_bad_out  = int((df.label == "bad").sum())

    print(f"\n{'=' * 60}")
    print(f"  Dataset saved → {output_path}")
    print(f"  Rows   : {len(df)}  (GOOD: {n_good_out}  |  BAD: {n_bad_out})")
    print(f"  Columns: {len(df.columns)}  ({len(kp_cols)} keypoint cols + {len(feat_cols)} derived)")
    print(f"  Skipped: {skipped} frames (no detection / missing file)")

    if n_good_out == 0 or n_bad_out == 0:
        print("\n  ⚠  WARNING: Only one class present — label more frames before training.")

    print(f"\n  Next step:  python train.py  (or use the CSV in your own notebook)")
    print("=" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate push-up keypoint dataset with YOLOv8 pose."
    )
    parser.add_argument(
        "--annotations", default="annotations.json",
        help="Path to annotations.json (default: annotations.json)",
    )
    parser.add_argument(
        "--output", default="dataset.csv",
        help="Output CSV path (default: dataset.csv)",
    )
    parser.add_argument(
        "--model", default="yolov8n-pose.pt",
        help="YOLOv8 pose model name or path (default: yolov8n-pose.pt). "
             "Larger options: yolov8s-pose.pt, yolov8m-pose.pt",
    )
    parser.add_argument(
        "--conf", type=float, default=0.30,
        help="Minimum YOLO detection confidence (default: 0.30)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.annotations):
        print(f"Annotations file not found: {args.annotations}")
        print("Run  python annotator.py  first, then try again.")
        sys.exit(1)

    generate(args.annotations, args.model, args.output, args.conf)


if __name__ == "__main__":
    main()
