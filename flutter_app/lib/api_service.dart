import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

class ApiService {
  // If running on Android emulator, use 10.0.2.2. For Web/iOS, localhost is fine if port forwarding is set mostly.
  // For web dev mode, localhost:8000 usually works directly.
  static const String baseUrl = 'http://127.0.0.1:8000';

  Future<Map<String, dynamic>> predictMaqam(
      Uint8List fileBytes, String filename) async {
    var uri = Uri.parse('$baseUrl/predict');
    var request = http.MultipartRequest('POST', uri);

    request.files.add(http.MultipartFile.fromBytes(
      'file',
      fileBytes,
      filename: filename,
      contentType: MediaType('audio', 'wav'),
    ));

    try {
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to analyze audio: \${response.body}');
      }
    } catch (e) {
      throw Exception('Error connecting to server: \$e');
    }
  }
}
