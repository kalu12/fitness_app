import 'package:flutter/foundation.dart' show kIsWeb;

// On web the app is served by the same Python server, so we use the same
// origin — no hardcoded IP needed, ngrok URL just works automatically.
// On Android native (emulator: 10.0.2.2, physical device: your PC's LAN IP).
String get kServerUrl {
  if (kIsWeb) return ''; // same-origin: relative URL
  return 'http://192.168.1.42:8000'; // ← change to your PC's IP for a physical device
}
