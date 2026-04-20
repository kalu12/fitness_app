#!/usr/bin/env python3
"""
Push-up Frame Annotator
=======================
Label video frames as GOOD (correct form), BAD (incorrect form), or INVALID (skip).

Usage:
  python annotator.py video1.mp4 video2.mp4 ...
  python annotator.py                          # opens file picker
  python annotator.py --every 5               # extract every 5th frame

Keyboard shortcuts:
  G           — Mark GOOD (correct push-up form)
  B           — Mark BAD (incorrect push-up form)
  I / Space   — Mark INVALID (blurry, transitional, not useful)
  ← / →       — Navigate without labeling
  Ctrl+S      — Save now (auto-saves on close too)
  Home / End  — Jump to first / last frame
  U           — Jump to next unlabeled frame
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk


# ─── Constants ────────────────────────────────────────────────────────────────

ANNOTATIONS_FILE = "annotations.json"
STATE_FILE = ".annotator_state.json"
FRAMES_DIR = "frames"

COLORS = {
    "good":    "#27ae60",
    "bad":     "#e74c3c",
    "invalid": "#636e72",
    None:      "#2d3436",
}

BG       = "#1e1e2e"
BG_TOP   = "#181825"
BG_MID   = "#1e1e2e"
FG       = "#cdd6f4"
FG_DIM   = "#6c7086"
ACCENT   = "#89b4fa"


# ─── Frame extraction ─────────────────────────────────────────────────────────

def extract_frames(video_path: str, output_dir: str, every_n: int = 3) -> list:
    """
    Extract every Nth frame from a video into output_dir.
    Skips frames that already exist on disk.
    Returns sorted list of all extracted frame paths for this video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_name = Path(video_path).stem
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(output_dir, exist_ok=True)

    saved     = []
    new_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            filename = f"{video_name}_f{frame_idx:06d}.jpg"
            path     = os.path.join(output_dir, filename)
            if not os.path.exists(path):
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                new_count += 1
            saved.append(path)

        frame_idx += 1

    cap.release()

    if new_count:
        print(f"  → Extracted {new_count} new frames  ({len(saved)} total from {Path(video_path).name})")
    else:
        print(f"  → {len(saved)} frames already on disk for {Path(video_path).name}  (skipped re-extraction)")

    return sorted(saved)


# ─── Persistence ──────────────────────────────────────────────────────────────

def load_annotations() -> dict:
    if os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE) as f:
            return json.load(f)
    return {}


def save_annotations(annotations: dict):
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations, f, indent=2, sort_keys=True)


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ─── GUI ──────────────────────────────────────────────────────────────────────

class AnnotatorApp:
    def __init__(self, root: tk.Tk, frames: list, annotations: dict):
        self.root        = root
        self.frames      = frames
        self.annotations = annotations
        self.idx         = 0
        self._photo      = None   # keep reference so GC doesn't kill it
        self._img_rgb    = None

        self.root.title("Push-up Annotator")
        self.root.configure(bg=BG)
        self.root.geometry("1100x780")
        self.root.minsize(700, 550)

        self._build_ui()
        self._bind_keys()

        # Restore last position
        state = load_state()
        last  = state.get("last_index", 0)
        self.idx = min(last, len(self.frames) - 1) if self.frames else 0

        self._refresh()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar (stats) ──────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG_TOP, pady=6)
        top.pack(fill=tk.X)

        self.lbl_stats = tk.Label(
            top, text="", bg=BG_TOP, fg=FG_DIM, font=("Courier", 10)
        )
        self.lbl_stats.pack()

        # ── Frame info bar ───────────────────────────────────────────────────
        info_bar = tk.Frame(self.root, bg=BG_MID, pady=4)
        info_bar.pack(fill=tk.X)

        self.lbl_info = tk.Label(
            info_bar, text="", bg=BG_MID, fg=ACCENT, font=("Courier", 12, "bold")
        )
        self.lbl_info.pack()

        # ── Canvas ───────────────────────────────────────────────────────────
        canvas_wrap = tk.Frame(self.root, bg=BG)
        canvas_wrap.pack(expand=True, fill=tk.BOTH, padx=10, pady=(4, 0))

        self.canvas = tk.Canvas(canvas_wrap, bg=BG, highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # ── Current label display ─────────────────────────────────────────────
        self.lbl_current = tk.Label(
            self.root, text="",
            font=("Courier", 18, "bold"),
            bg=BG, fg=FG_DIM, pady=4,
        )
        self.lbl_current.pack()

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = tk.Frame(self.root, bg=BG, pady=6)
        btn_row.pack()

        buttons = [
            ("◀  PREV\n[←]",        self._prev,                       "#3d3d5c", FG),
            ("✓  GOOD\n[G]",         lambda: self._label_and_advance("good"),    "#1d6336", "#a6e3a1"),
            ("✗  BAD\n[B]",          lambda: self._label_and_advance("bad"),     "#6b1d1d", "#f38ba8"),
            ("⊘  INVALID\n[I/Space]",lambda: self._label_and_advance("invalid"), "#3a3a4a", "#a6adc8"),
            ("NEXT  ▶\n[→]",         self._next,                       "#3d3d5c", FG),
        ]

        for text, cmd, bg, fg in buttons:
            b = tk.Button(
                btn_row, text=text, command=cmd,
                bg=bg, fg=fg, relief=tk.FLAT,
                font=("Courier", 10, "bold"),
                width=13, height=3,
                cursor="hand2",
                activebackground=bg, activeforeground=fg,
            )
            b.pack(side=tk.LEFT, padx=6)

        # ── Progress bar ─────────────────────────────────────────────────────
        prog_wrap = tk.Frame(self.root, bg=BG, padx=12, pady=4)
        prog_wrap.pack(fill=tk.X)

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "flat.Horizontal.TProgressbar",
            troughcolor="#313244",
            background="#89b4fa",
            bordercolor=BG,
            lightcolor="#89b4fa",
            darkcolor="#89b4fa",
        )

        self.progress = ttk.Progressbar(
            prog_wrap, mode="determinate",
            style="flat.Horizontal.TProgressbar",
        )
        self.progress.pack(fill=tk.X)

        self.lbl_progress = tk.Label(
            self.root, text="", bg=BG, fg=FG_DIM, font=("Courier", 9), pady=3
        )
        self.lbl_progress.pack()

    # ── Key bindings ──────────────────────────────────────────────────────────

    def _bind_keys(self):
        for key in ("g", "G"):
            self.root.bind(f"<{key}>", lambda e: self._label_and_advance("good"))
        for key in ("b", "B"):
            self.root.bind(f"<{key}>", lambda e: self._label_and_advance("bad"))
        for key in ("i", "I"):
            self.root.bind(f"<{key}>", lambda e: self._label_and_advance("invalid"))
        self.root.bind("<space>",     lambda e: self._label_and_advance("invalid"))
        self.root.bind("<Left>",      lambda e: self._prev())
        self.root.bind("<Right>",     lambda e: self._next())
        self.root.bind("<Home>",      lambda e: self._goto(0))
        self.root.bind("<End>",       lambda e: self._goto(len(self.frames) - 1))
        self.root.bind("<u>",         lambda e: self._goto_unlabeled())
        self.root.bind("<U>",         lambda e: self._goto_unlabeled())
        self.root.bind("<Control-s>", lambda e: self._save_manual())
        self.root.bind("<Configure>", lambda e: self._redraw())

    # ── Navigation ────────────────────────────────────────────────────────────

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._refresh()

    def _next(self):
        if self.idx < len(self.frames) - 1:
            self.idx += 1
            self._refresh()

    def _goto(self, idx: int):
        self.idx = max(0, min(idx, len(self.frames) - 1))
        self._refresh()

    def _goto_unlabeled(self):
        # search forward first, then wrap
        for offset in range(1, len(self.frames)):
            i = (self.idx + offset) % len(self.frames)
            if self.frames[i] not in self.annotations:
                self.idx = i
                self._refresh()
                return
        messagebox.showinfo("Done!", "All frames have been labeled.")

    # ── Labeling ──────────────────────────────────────────────────────────────

    def _label_and_advance(self, label: str):
        if not self.frames:
            return
        self.annotations[self.frames[self.idx]] = label
        save_annotations(self.annotations)
        self._refresh()
        self._next()

    # ── Display ───────────────────────────────────────────────────────────────

    def _refresh(self):
        """Reload image and update all UI elements."""
        if not self.frames:
            return
        path = self.frames[self.idx]
        img  = cv2.imread(path)
        if img is not None:
            self._img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._redraw()
        self._update_labels()
        save_state({"last_index": self.idx})

    def _redraw(self):
        """Repaint canvas with current image (called on resize too)."""
        if self._img_rgb is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 960, 500

        h, w  = self._img_rgb.shape[:2]
        scale = min(cw / w, ch / h)
        nw    = int(w * scale)
        nh    = int(h * scale)

        resized     = cv2.resize(self._img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        pil_img     = Image.fromarray(resized)
        self._photo = ImageTk.PhotoImage(pil_img)

        cx = cw // 2
        cy = ch // 2

        self.canvas.delete("all")
        self.canvas.create_image(cx, cy, anchor=tk.CENTER, image=self._photo)

        # Colored border showing current label
        label = self.annotations.get(self.frames[self.idx]) if self.frames else None
        if label:
            border_color = COLORS.get(label, "#ffffff")
            x0 = cx - nw // 2
            y0 = cy - nh // 2
            self.canvas.create_rectangle(
                x0, y0, x0 + nw, y0 + nh,
                outline=border_color, width=5,
            )

    def _update_labels(self):
        if not self.frames:
            return

        path  = self.frames[self.idx]
        label = self.annotations.get(path)

        # Info bar
        fname = Path(path).name
        self.lbl_info.config(
            text=f"Frame {self.idx + 1} / {len(self.frames)}   |   {fname}"
        )

        # Current label chip
        text_map = {
            "good":    "✓  GOOD — Correct push-up form",
            "bad":     "✗  BAD — Incorrect form",
            "invalid": "⊘  INVALID — Skip this frame",
            None:      "—  UNLABELED",
        }
        self.lbl_current.config(
            text=COLORS.get(label, FG_DIM) and text_map.get(label, "—  UNLABELED"),
            fg=COLORS.get(label, FG_DIM),
        )
        # re-apply text separately since we abused the dict lookup above
        self.lbl_current.config(text=text_map.get(label, "—  UNLABELED"))

        # Stats
        n_good    = sum(1 for v in self.annotations.values() if v == "good")
        n_bad     = sum(1 for v in self.annotations.values() if v == "bad")
        n_inv     = sum(1 for v in self.annotations.values() if v == "invalid")
        n_labeled = n_good + n_bad + n_inv
        n_total   = len(self.frames)

        self.lbl_stats.config(
            text=(
                f"✓ GOOD: {n_good}   "
                f"✗ BAD: {n_bad}   "
                f"⊘ INVALID: {n_inv}   "
                f"— UNLABELED: {n_total - n_labeled}   "
                f"TOTAL: {n_total}"
            )
        )

        pct = n_labeled / n_total * 100 if n_total else 0
        self.progress["value"] = pct
        self.lbl_progress.config(text=f"{n_labeled} / {n_total} labeled  ({pct:.1f}%)")

        # Label color on current-label widget
        self.lbl_current.config(fg=COLORS.get(label, FG_DIM))

    # ── Save / close ──────────────────────────────────────────────────────────

    def _save_manual(self):
        save_annotations(self.annotations)
        self.lbl_stats.config(text="  ✔ Saved!  ")
        self.root.after(1500, lambda: self._update_labels())

    def on_close(self):
        save_annotations(self.annotations)
        self.root.destroy()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Annotate push-up video frames as GOOD / BAD / INVALID."
    )
    parser.add_argument("videos", nargs="*", help="Video file paths to annotate")
    parser.add_argument(
        "--every", type=int, default=3, metavar="N",
        help="Extract every Nth frame (default: 3  ≈ 10fps for 30fps video)",
    )
    parser.add_argument(
        "--frames-dir", default=FRAMES_DIR,
        help=f"Folder for extracted frames (default: {FRAMES_DIR}/)",
    )
    args = parser.parse_args()

    # ── Get video paths ───────────────────────────────────────────────────────
    videos = list(args.videos)
    if not videos:
        tmp = tk.Tk()
        tmp.withdraw()
        selected = filedialog.askopenfilenames(
            title="Select push-up video files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV")],
        )
        tmp.destroy()
        if not selected:
            print("No videos selected. Exiting.")
            sys.exit(0)
        videos = list(selected)

    # ── Extract frames ────────────────────────────────────────────────────────
    all_frames = []
    for video in sorted(videos):
        print(f"Processing {video} …")
        try:
            frames = extract_frames(video, args.frames_dir, every_n=args.every)
            all_frames.extend(frames)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    all_frames = sorted(set(all_frames))

    if not all_frames:
        print("No frames found. Exiting.")
        sys.exit(1)

    annotations = load_annotations()
    n_existing  = sum(1 for f in all_frames if f in annotations)
    print(f"\n{len(all_frames)} frames total  |  {n_existing} already labeled")
    print("Launching annotator…\n")

    # ── Launch GUI ────────────────────────────────────────────────────────────
    root = tk.Tk()
    app  = AnnotatorApp(root, all_frames, annotations)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
