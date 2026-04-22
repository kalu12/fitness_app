import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../config.dart';
import '../models/prediction.dart';

class ApiService {
  static Future<PredictionResult> predict(Uint8List jpegBytes) async {
    final base = kServerUrl;
    final uri = base.isEmpty ? Uri(path: '/predict') : Uri.parse('$base/predict');

    final request = http.MultipartRequest('POST', uri)
      ..files.add(http.MultipartFile.fromBytes(
        'file',
        jpegBytes,
        filename: 'frame.jpg',
      ));

    final streamed = await request.send().timeout(const Duration(seconds: 5));
    final body = await streamed.stream.bytesToString();

    if (streamed.statusCode != 200) {
      throw Exception('Server error ${streamed.statusCode}: $body');
    }

    final json = jsonDecode(body) as Map<String, dynamic>;
    if (json.containsKey('error')) {
      throw Exception(json['error']);
    }

    return PredictionResult.fromJson(json);
  }

  static Future<bool> isHealthy() async {
    try {
      final base = kServerUrl;
      final uri = base.isEmpty ? Uri(path: '/health') : Uri.parse('$base/health');
      final res = await http.get(uri).timeout(const Duration(seconds: 3));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
