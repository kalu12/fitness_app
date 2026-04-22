class Keypoints {
  final List<List<double>> xyNorm; // 17 × [x, y] in [0,1]
  final List<double> kpConf;       // 17 confidence values

  const Keypoints({required this.xyNorm, required this.kpConf});

  factory Keypoints.fromJson(Map<String, dynamic> json) {
    final raw = json['xy_norm'] as List;
    final xyNorm = raw
        .map<List<double>>((pt) => (pt as List).map<double>((v) => (v as num).toDouble()).toList())
        .toList();
    final kpConf =
        (json['kp_conf'] as List).map<double>((v) => (v as num).toDouble()).toList();
    return Keypoints(xyNorm: xyNorm, kpConf: kpConf);
  }
}

class PredictionResult {
  final String label;          // "good" | "bad" | "no_detection"
  final double confidence;
  final List<String> feedback;
  final Keypoints? keypoints;

  const PredictionResult({
    required this.label,
    required this.confidence,
    required this.feedback,
    this.keypoints,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      label: json['label'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      feedback: (json['feedback'] as List).cast<String>(),
      keypoints: json['keypoints'] != null
          ? Keypoints.fromJson(json['keypoints'] as Map<String, dynamic>)
          : null,
    );
  }
}
