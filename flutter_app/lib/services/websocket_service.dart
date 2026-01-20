import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:web_socket_channel/web_socket_channel.dart';

class WebSocketService {
  WebSocketChannel? _channel;
  final StreamController<Map<String, dynamic>> _dataController =
      StreamController.broadcast();

  Stream<Map<String, dynamic>> get dataStream => _dataController.stream;

  void connect() {
    // For Android Emulator use '10.0.2.2'. For Web/iOS use '127.0.0.1' or 'localhost'.
    // Adjust port if needed.
    final uri = Uri.parse('ws://127.0.0.1:8000/ws/analyze');
    _channel = WebSocketChannel.connect(uri);
    print("Connecting to WS...");

    _channel!.stream.listen(
      (message) {
        try {
          final decoded = jsonDecode(message);
          _dataController.add(decoded);
        } catch (e) {
          print("Error decoding JSON: $e");
        }
      },
      onError: (error) => print("WS Error: $error"),
      onDone: () => print("WS Closed"),
    );
  }

  void sendAudioChunk(Uint8List bytes) {
    if (_channel != null) {
      _channel!.sink.add(bytes);
    }
  }

  void disconnect() {
    _channel?.sink.close();
  }
}
