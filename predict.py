#!/usr/bin/env python3
"""
Push-up Form Predictor
=======================
Loads the trained model (model.pkl) + YOLOv8 pose and runs inference
on a single image, a folder of images, or a live video/webcam feed.

Usage:
  python predict.py frame.jpg                   # single image
  python predict.py frames/                     # all images in folder
  python predict.py video.mp4                   # annotate every frame
  python predict.py --webcam                    # live webcam (press Q to quit)
"""

import argparse
import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed.  Run: pip install ultralytics")
    sys.exit(1)

# Reuse feature extraction from generate_dataset
sys.path.insert(0, os.path.dirname(__file__))
from generate_dataset import extract_features, KP_NAMES


# ─── Colour palette ───────────────────────────────────────────────────────────
GREEN  = (39,  174,  96)
RED    = (231,  76,  60)
YELLOW = (241, 196,  15)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)


# ─── Load model ───────────────────────────────────────────────────────────────

def load_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"ERROR: model file '{model_path}' not found.")
        print("Run  python train.py  first.")
        sys.exit(1)
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    print(f"Model loaded: {payload['model_name']}  "
          f"(CV F1-macro={payload['cv_results'][payload['model_name']]['f1_macro']:.3f}  "
          f"AUC={payload['cv_results'][payload['model_name']]['auc']:.3f})")
    return payload["model"], payload["feature_cols"]


# ─── Single-frame inference ───────────────────────────────────────────────────

def predict_frame(img_bgr: np.ndarray, pose_model, clf, feature_cols: list,
                  conf_threshold: float = 0.30):
    """
    Returns:
      label      : "good" | "bad" | "no_detection"
      confidence : float  probability of predicted class
      feedback   : list[str]  improvement cues (empty when label == "good")
      kp_data    : dict with xy_norm and conf arrays for visualisation
    """
    results = pose_model(img_bgr, verbose=False, conf=conf_threshold)

    if not results or results[0].keypoints is None or len(results[0].boxes) == 0:
        return "no_detection", 0.0, ["Could not detect a person in the frame."], {}

    r    = results[0]
    best = int(r.boxes.conf.argmax())
    kps  = r.keypoints

    xy_px   = kps.xy[best].cpu().numpy()        # (17, 2)
    kp_conf = kps.conf[best].cpu().numpy()      # (17,)

    h, w    = img_bgr.shape[:2]
    xy_norm = xy_px / np.array([w, h])

    feats   = extract_features(xy_norm, kp_conf)

    # Build feature vector in exact training order
    import pandas as pd
    row = pd.DataFrame([feats])
    # Replace sentinel missing values with NaN
    row.replace([-1.0, -999.0], np.nan, inplace=True)

    # Keep only columns the model was trained on; fill unknown cols with NaN
    X = row.reindex(columns=feature_cols)

    prob = clf.predict_proba(X)[0]              # [p_bad, p_good]
    pred_int  = int(clf.predict(X)[0])
    label     = "good" if pred_int == 1 else "bad"
    confidence = float(prob[pred_int])

    # Generate improvement feedback
    from train import generate_feedback
    feedback = generate_feedback(feats) if label == "bad" else []

    return label, confidence, feedback, {
        "xy_px": xy_px, "xy_norm": xy_norm, "kp_conf": kp_conf
    }


# ─── Visualisation ────────────────────────────────────────────────────────────

SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),      # legs
    (0, 5), (0, 6),                              # head–shoulder
]


def draw_overlay(img: np.ndarray, label: str, confidence: float,
                 feedback: list, kp_data: dict) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    color = GREEN if label == "good" else RED

    # ── Skeleton ──────────────────────────────────────────────────────────────
    if kp_data:
        xy  = kp_data["xy_px"].astype(int)
        kpc = kp_data["kp_conf"]

        for a, b in SKELETON_PAIRS:
            if kpc[a] > 0.3 and kpc[b] > 0.3:
                cv2.line(out, tuple(xy[a]), tuple(xy[b]), color, 2, cv2.LINE_AA)

        for i, (x, y_) in enumerate(xy):
            if kpc[i] > 0.3:
                cv2.circle(out, (x, y_), 5, WHITE, -1)
                cv2.circle(out, (x, y_), 5, color,  2)

    # ── Label banner ──────────────────────────────────────────────────────────
    text    = f"{'GOOD' if label == 'good' else 'BAD'}  {confidence*100:.0f}%"
    font    = cv2.FONT_HERSHEY_DUPLEX
    scale   = 1.1
    thick   = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad     = 10
    cv2.rectangle(out, (0, 0), (tw + pad*2, th + pad*2 + 4), color, -1)
    cv2.putText(out, text, (pad, th + pad), font, scale, WHITE, thick, cv2.LINE_AA)

    # ── Feedback tips ─────────────────────────────────────────────────────────
    if feedback:
        tip_font  = cv2.FONT_HERSHEY_SIMPLEX
        tip_scale = 0.52
        tip_thick = 1
        line_h    = 22
        y_start   = h - (len(feedback) * line_h) - 10
        bg_h      = len(feedback) * line_h + 16
        cv2.rectangle(out, (0, h - bg_h), (w, h), (20, 20, 20), -1)
        for i, tip in enumerate(feedback):
            y_pos = y_start + i * line_h
            cv2.putText(out, f"• {tip}", (10, y_pos),
                        tip_font, tip_scale, YELLOW, tip_thick, cv2.LINE_AA)

    return out


# ─── Modes ────────────────────────────────────────────────────────────────────

def run_image(path: str, pose_model, clf, feature_cols, save_dir: str):
    img = cv2.imread(path)
    if img is None:
        print(f"  Cannot read: {path}")
        return

    label, conf, feedback, kp_data = predict_frame(img, pose_model, clf, feature_cols)
    vis = draw_overlay(img, label, conf, feedback, kp_data)

    out_name = Path(path).stem + "_predicted.jpg"
    out_path = os.path.join(save_dir, out_name)
    cv2.imwrite(out_path, vis)

    print(f"  {Path(path).name:40s}  →  {label.upper():4s}  ({conf*100:.0f}%)", end="")
    if feedback:
        print(f"  Tips: {len(feedback)}")
        for tip in feedback:
            print(f"    • {tip}")
    else:
        print()


def run_folder(folder: str, pose_model, clf, feature_cols):
    exts    = {".jpg", ".jpeg", ".png", ".bmp"}
    images  = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)
    out_dir = os.path.join(folder, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Processing {len(images)} images in {folder}  →  {out_dir}")
    for img_path in images:
        run_image(str(img_path), pose_model, clf, feature_cols, out_dir)


def run_video(video_path: str, pose_model, clf, feature_cols, every_n: int = 1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(video_path).stem + "_predicted.mp4"
    writer   = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / every_n,
        (w, h_),
    )

    print(f"Processing {total} frames  (every {every_n})  →  {out_path}")

    last_label, last_conf, last_fb, last_kp = "no_detection", 0.0, [], {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            last_label, last_conf, last_fb, last_kp = predict_frame(
                frame, pose_model, clf, feature_cols
            )

        vis = draw_overlay(frame, last_label, last_conf, last_fb, last_kp)
        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved → {out_path}")


def run_webcam(pose_model, clf, feature_cols, camera_idx: int = 0):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_idx}")
        return

    print("Live webcam — press  Q  to quit,  S  to save screenshot")
    frame_n   = 0
    last_pred = ("no_detection", 0.0, [], {})

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference every 3rd frame for smooth display
        if frame_n % 3 == 0:
            last_pred = predict_frame(frame, pose_model, clf, feature_cols)

        vis = draw_overlay(frame, *last_pred[:3], last_pred[3])
        cv2.imshow("Push-up Form Checker  (Q=quit  S=save)", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"snapshot_{frame_n:05d}.jpg"
            cv2.imwrite(fname, vis)
            print(f"Saved snapshot → {fname}")

        frame_n += 1

    cap.release()
    cv2.destroyAllWindows()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict push-up form quality.")
    parser.add_argument("input",   nargs="?",  help="Image / folder / video path")
    parser.add_argument("--model", default="model.pkl",       help="Trained model file")
    parser.add_argument("--pose",  default="yolov8n-pose.pt", help="YOLOv8 pose weights")
    parser.add_argument("--webcam",action="store_true",       help="Use live webcam")
    parser.add_argument("--every", type=int, default=3,
                        help="For video: process every Nth frame (default 3)")
    parser.add_argument("--save-dir", default="predictions",
                        help="Output folder for single-image predictions")
    args = parser.parse_args()

    if not args.webcam and not args.input:
        parser.print_help()
        sys.exit(0)

    clf, feature_cols = load_model(args.model)

    print(f"Loading pose model '{args.pose}' …")
    pose_model = YOLO(args.pose)

    if args.webcam:
        run_webcam(pose_model, clf, feature_cols)

    elif os.path.isdir(args.input):
        run_folder(args.input, pose_model, clf, feature_cols)

    elif args.input.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        run_video(args.input, pose_model, clf, feature_cols, every_n=args.every)

    else:
        os.makedirs(args.save_dir, exist_ok=True)
        run_image(args.input, pose_model, clf, feature_cols, args.save_dir)


if __name__ == "__main__":
    main()
