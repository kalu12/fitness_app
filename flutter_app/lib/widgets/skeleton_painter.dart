import 'package:flutter/material.dart';
import '../models/prediction.dart';

// Mirrors SKELETON_PAIRS from predict.py
const _pairs = [
  [5, 6],  [5, 7],  [7, 9],  [6, 8],  [8, 10], // arms
  [5, 11], [6, 12], [11, 12],                    // torso
  [11, 13],[13, 15],[12, 14], [14, 16],           // legs
  [0, 5],  [0, 6],                               // head–shoulder
];

class SkeletonPainter extends CustomPainter {
  final PredictionResult prediction;

  // Intrinsic size of the image that was sent to the server.
  // We use this to correctly account for aspect-ratio letterboxing inside
  // the camera preview widget.
  final Size imageSize;

  const SkeletonPainter({required this.prediction, required this.imageSize});

  @override
  void paint(Canvas canvas, Size widgetSize) {
    final kps = prediction.keypoints;
    if (kps == null) return;

    final isGood = prediction.label == 'good';
    final color = isGood ? const Color(0xFF27AE60) : const Color(0xFFE74C3C);

    // ── Compute the letterbox rect ─────────────────────────────────────────────
    // CameraPreview uses BoxFit.cover, which fills the widget and may crop the
    // sides or top/bottom.  We approximate by using BoxFit.contain math so the
    // skeleton lines line up with the visible joints.
    //
    // In practice push-up detection works with the subject centred, so even a
    // slight mismatch is acceptable.
    final double imgAR = imageSize.width / imageSize.height;
    final double widAR = widgetSize.width / widgetSize.height;

    double scaleX, scaleY, offsetX, offsetY;
    if (imgAR > widAR) {
      // Image wider than widget → pillarbox (left/right bars)
      scaleX = widgetSize.width / imageSize.width;
      scaleY = scaleX;
      offsetX = 0;
      offsetY = (widgetSize.height - imageSize.height * scaleY) / 2;
    } else {
      // Image taller than widget → letterbox (top/bottom bars)
      scaleY = widgetSize.height / imageSize.height;
      scaleX = scaleY;
      offsetX = (widgetSize.width - imageSize.width * scaleX) / 2;
      offsetY = 0;
    }

    Offset toScreen(double nx, double ny) => Offset(
          nx * imageSize.width * scaleX + offsetX,
          ny * imageSize.height * scaleY + offsetY,
        );

    final linePaint = Paint()
      ..color = color
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;

    final dotFill = Paint()..color = Colors.white;
    final dotBorder = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    // Skeleton lines
    for (final pair in _pairs) {
      final a = pair[0], b = pair[1];
      if (kps.kpConf[a] > 0.3 && kps.kpConf[b] > 0.3) {
        canvas.drawLine(
          toScreen(kps.xyNorm[a][0], kps.xyNorm[a][1]),
          toScreen(kps.xyNorm[b][0], kps.xyNorm[b][1]),
          linePaint,
        );
      }
    }

    // Joint dots
    for (int i = 0; i < kps.xyNorm.length; i++) {
      if (kps.kpConf[i] > 0.3) {
        final pos = toScreen(kps.xyNorm[i][0], kps.xyNorm[i][1]);
        canvas.drawCircle(pos, 5, dotFill);
        canvas.drawCircle(pos, 5, dotBorder);
      }
    }
  }

  @override
  bool shouldRepaint(covariant SkeletonPainter old) =>
      old.prediction != prediction || old.imageSize != imageSize;
}
