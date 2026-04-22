import 'dart:async';
import 'dart:io' show File;
import 'dart:ui' as ui;
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import '../models/prediction.dart';
import '../services/api_service.dart';
import '../widgets/skeleton_painter.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  bool _isProcessing = false;
  bool _serverReachable = false;
  PredictionResult? _prediction;
  Size _imageSize = const Size(640, 480);
  Timer? _predictionTimer;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _checkServer();
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _predictionTimer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _predictionTimer?.cancel();
      ctrl.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _checkServer() async {
    final ok = await ApiService.isHealthy();
    if (mounted) setState(() => _serverReachable = ok);
    // Keep polling until reachable
    if (!ok) {
      Future.delayed(const Duration(seconds: 3), _checkServer);
    }
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _error = 'No camera found on this device.');
        return;
      }

      // Prefer front camera for push-up detection
      final cam = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      final ctrl = CameraController(
        cam,
        ResolutionPreset.medium, // 640×480 — good balance of speed vs quality
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await ctrl.initialize();
      if (!mounted) return;

      setState(() {
        _controller = ctrl;
        _error = null;
      });

      // Send a frame every 500 ms
      _predictionTimer = Timer.periodic(
        const Duration(milliseconds: 500),
        (_) => _captureAndPredict(),
      );
    } catch (e) {
      if (mounted) setState(() => _error = 'Camera error: $e');
    }
  }

  Future<void> _captureAndPredict() async {
    if (_isProcessing) return;
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) return;
    if (!_serverReachable) return;

    _isProcessing = true;
    try {
      final xfile = await ctrl.takePicture();
      // xfile.readAsBytes() works on both native and web
      final bytes = await xfile.readAsBytes();

      // Decode image dimensions for accurate skeleton overlay
      final decoded = await decodeImageFromList(bytes);
      final imgSize = Size(
        decoded.width.toDouble(),
        decoded.height.toDouble(),
      );

      final result = await ApiService.predict(bytes);

      if (mounted) {
        setState(() {
          _prediction = result;
          _imageSize = imgSize;
          _serverReachable = true;
        });
      }

      // Clean up temp file on native platforms only
      if (!kIsWeb) {
        await File(xfile.path).delete().catchError((_) {});
      }
    } catch (_) {
      // Silently ignore individual frame errors to keep the loop running
    } finally {
      _isProcessing = false;
    }
  }

  // ── UI helpers ─────────────────────────────────────────────────────────────

  Color get _statusColor {
    if (_prediction == null) return Colors.grey;
    return _prediction!.label == 'good'
        ? const Color(0xFF27AE60)
        : const Color(0xFFE74C3C);
  }

  String get _statusText {
    if (_prediction == null) return '';
    if (_prediction!.label == 'no_detection') return '';
    final pct = (_prediction!.confidence * 100).toStringAsFixed(0);
    return '${_prediction!.label == 'good' ? 'GOOD' : 'BAD'}  $pct%';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(child: _buildBody()),
    );
  }

  Widget _buildBody() {
    if (_error != null) return _buildError(_error!);

    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: Colors.white),
            SizedBox(height: 16),
            Text('Initialising camera…',
                style: TextStyle(color: Colors.white70)),
          ],
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        // ── Camera preview ──────────────────────────────────────────────────
        CameraPreview(ctrl),

        // ── Skeleton overlay ────────────────────────────────────────────────
        if (_prediction?.keypoints != null)
          LayoutBuilder(builder: (ctx, constraints) {
            return CustomPaint(
              size: Size(constraints.maxWidth, constraints.maxHeight),
              painter: SkeletonPainter(
                prediction: _prediction!,
                imageSize: _imageSize,
              ),
            );
          }),

        // ── Label banner (top-left) ─────────────────────────────────────────
        if (_statusText.isNotEmpty)
          Positioned(
            top: 0,
            left: 0,
            child: Container(
              color: _statusColor,
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Text(
                _statusText,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.2,
                ),
              ),
            ),
          ),

        // ── Server status indicator (top-right) ─────────────────────────────
        Positioned(
          top: 8,
          right: 12,
          child: Row(
            children: [
              Icon(
                Icons.circle,
                size: 10,
                color: _serverReachable ? Colors.greenAccent : Colors.red,
              ),
              const SizedBox(width: 4),
              Text(
                _serverReachable ? 'Connected' : 'No server',
                style: TextStyle(
                  color: _serverReachable ? Colors.greenAccent : Colors.red,
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),

        // ── Feedback tips (bottom) ──────────────────────────────────────────
        if (_prediction != null && _prediction!.feedback.isNotEmpty)
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              color: const Color(0xDD141414),
              padding: const EdgeInsets.fromLTRB(12, 10, 12, 14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: _prediction!.feedback
                    .map((tip) => Padding(
                          padding: const EdgeInsets.symmetric(vertical: 2),
                          child: Text(
                            '• $tip',
                            style: const TextStyle(
                              color: Color(0xFFF1C40F),
                              fontSize: 13,
                            ),
                          ),
                        ))
                    .toList(),
              ),
            ),
          ),

        // ── No server warning ───────────────────────────────────────────────
        if (!_serverReachable)
          Positioned(
            bottom: 24,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 24),
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.black87,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.orange),
                ),
                child: const Text(
                  'Start the Python server:\npython server.py',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.orange, fontSize: 13),
                ),
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildError(String msg) => Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            msg,
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.redAccent, fontSize: 15),
          ),
        ),
      );
}
