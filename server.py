"""
Push-up Form Checker — FastAPI server
Wraps predict.py so the Flutter app can send camera frames and get predictions back.

Run:
    pip install fastapi uvicorn python-multipart
    python server.py

The server listens on 0.0.0.0:8000 and also serves the Flutter web build.

iPhone testing (requires HTTPS for camera access):
    1. flutter build web            (inside flutter_app/)
    2. python server.py             (serves API + Flutter web app)
    3. ngrok http 8000              (creates public HTTPS URL)
    4. Open the ngrok URL in iPhone Safari
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(__file__))

app = FastAPI(title="Fitness Form Checker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup
_pose_model = None
_clf = None
_feature_cols = None


@app.on_event("startup")
async def _startup():
    global _pose_model, _clf, _feature_cols
    from predict import load_model
    from ultralytics import YOLO

    _clf, _feature_cols = load_model("model.pkl")

    print("Loading YOLOv8 pose model …")
    _pose_model = YOLO("yolov8n-pose.pt")
    print("Server ready — listening on http://0.0.0.0:8000")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _clf is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a JPEG/PNG image, returns:
      label      : "good" | "bad" | "no_detection"
      confidence : float [0..1]
      feedback   : list[str]
      keypoints  : { xy_norm: [[x,y]…], kp_conf: [c…] } | null
    """
    try:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"error": "Cannot decode image"}, status_code=400)

        from predict import predict_frame
        label, confidence, feedback, kp_data = predict_frame(
            frame, _pose_model, _clf, _feature_cols
        )

        kp_json = None
        if kp_data and "xy_norm" in kp_data:
            kp_json = {
                "xy_norm": kp_data["xy_norm"].tolist(),   # [[x,y], …] 17 points
                "kp_conf": kp_data["kp_conf"].tolist(),   # [c, …] 17 values
            }

        return {
            "label": label,
            "confidence": round(float(confidence), 3),
            "feedback": feedback,
            "keypoints": kp_json,
        }

    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# Serve the Flutter web build at "/" if it has been built.
# flutter build web  →  flutter_app/build/web/
_WEB_BUILD = os.path.join(os.path.dirname(__file__), "flutter_app", "build", "web")
if os.path.isdir(_WEB_BUILD):
    from fastapi.responses import FileResponse

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(_WEB_BUILD, "index.html"))

    app.mount("/", StaticFiles(directory=_WEB_BUILD), name="web")
    print(f"Serving Flutter web app from {_WEB_BUILD}")
else:
    print("Flutter web build not found. Run:  flutter build web  inside flutter_app/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
