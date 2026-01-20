import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import '../services/websocket_service.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final WebSocketService _wsService = WebSocketService();
  String _status = "Disconnected";
  String _currentMaqam = "Unknown";
  Map<String, dynamic> _scores = {};
  List<double> _chromagram = List.filled(36, 0.0);
  int _rukooz = -1;

  @override
  void initState() {
    super.initState();
    _connect();
  }

  void _connect() {
    _wsService.connect();
    setState(() => _status = "Connected");
    _wsService.dataStream.listen((data) {
      if (data.containsKey("chromagram")) {
        setState(() {
          // Flatten chromagram if it comes as [36][Time]
          // For simplicity, we just take the mean of the first frame or last frame
          // API sends [36, Time] list.
          List<dynamic> rawChroma = data["chromagram"];
          // We expect a list of 36 floats (if 1 frame) or list of lists
          // Let's assume the API sums it up or we simplify. 
          // If API sends 2D array, let's take the average energy per bin.
          
          if (rawChroma.isNotEmpty && rawChroma[0] is List) {
             // 2D case
             _chromagram = List.generate(36, (i) {
               double sum = 0;
               for(var frame in rawChroma) {
                 sum += (frame[i] as num).toDouble();
               }
               return sum / rawChroma.length; // This logic depends on exact shape
             });
          } else {
             // 1D case (flattened or single frame)
             _chromagram = rawChroma.map((e) => (e as num).toDouble()).toList();
          }

          _currentMaqam = data["prediction"] ?? "Unknown";
          _scores = data["confidence"] ?? {};
          _rukooz = data["rukooz"] ?? -1;
        });
      }
    });
  }

  Future<void> _uploadFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['wav'],
      withData: true,
    );

    if (result != null && result.files.single.bytes != null) {
      Uint8List fileBytes = result.files.single.bytes!;
      _wsService.sendAudioChunk(fileBytes);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Audio sent for analysis...")),
      );
    }
  }

  @override
  void dispose() {
    _wsService.disconnect();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("MAQAM DETECTOR 2.0"),
        centerTitle: true,
        actions: [
          Container(
            margin: const EdgeInsets.only(right: 16),
            child: CircleAvatar(
              radius: 6,
              backgroundColor: _status == "Connected" ? Colors.green : Colors.red,
            ),
          )
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Top Section: Prediction Card
            _buildPredictionCard(),
            const SizedBox(height: 20),
            
            // Middle: 36-Bin Chromagram Visualizer
            Expanded(
              flex: 2,
              child: _buildChromagramCircle(),
            ),
            
            const SizedBox(height: 20),
            
            // Bottom Controls
            ElevatedButton.icon(
              onPressed: _uploadFile,
              icon: const Icon(Icons.upload_file),
              label: const Text("UPLOAD WAV CHUNK"),
              style: ElevatedButton.styleFrom(
                backgroundColor: Theme.of(context).primaryColor,
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPredictionCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Theme.of(context).primaryColor.withOpacity(0.3)),
        boxShadow: [
          BoxShadow(color: Colors.black.withOpacity(0.5), blurRadius: 10, offset: const Offset(0, 4))
        ],
      ),
      child: Column(
        children: [
          const Text("DETECTED MAQAM", style: TextStyle(color: Colors.grey, fontSize: 12)),
          const SizedBox(height: 8),
          Text(
            _currentMaqam.toUpperCase(),
            style: TextStyle(
              color: Theme.of(context).colorScheme.secondary,
              fontSize: 36,
              fontWeight: FontWeight.bold,
              letterSpacing: 2,
            ),
          ),
          const SizedBox(height: 16),
          // Scores
          Wrap(
            spacing: 12,
            runSpacing: 12,
            alignment: WrapAlignment.center,
            children: _scores.entries.map((e) {
              return Chip(
                label: Text("${e.key}: ${(e.value as double).toStringAsFixed(2)}"),
                backgroundColor: Colors.black54,
                labelStyle: const TextStyle(fontSize: 10, color: Colors.white70),
              );
            }).toList(),
          )
        ],
      ),
    );
  }

  Widget _buildChromagramCircle() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.black,
        shape: BoxShape.circle,
        border: Border.all(color: Colors.white10),
      ),
      child: CustomPaint(
        painter: ChromagramPainter(_chromagram, _rukooz),
        child: Container(),
      ),
    );
  }
}

class ChromagramPainter extends CustomPainter {
  final List<double> energy;
  final int rukoozBin;

  ChromagramPainter(this.energy, this.rukoozBin);

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = min(size.width, size.height) / 2 - 20;
    final paint = Paint()..style = PaintingStyle.fill;
    
    // Draw 36 segments
    final anglePerBin = 2 * pi / 36;
    
    // Normalize energy for visualization
    double maxE = energy.reduce(max);
    if (maxE == 0) maxE = 1;

    for (int i = 0; i < 36; i++) {
      double binEnergy = (i < energy.length) ? energy[i] : 0;
      double barHeight = (binEnergy / maxE) * (radius * 0.8);
      
      final angle = i * anglePerBin - pi / 2; // Start from top
      
      paint.color = (i == rukoozBin) 
          ? const Color(0xFFFF0055) // Highlight Rukooz
          : const Color(0xFF00E5FF).withOpacity(0.6 + (binEnergy/maxE)*0.4);

      // Draw bars radiating outward
      final p1 = Offset(
        center.dx + cos(angle) * (radius * 0.2),
        center.dy + sin(angle) * (radius * 0.2),
      );
      final p2 = Offset(
        center.dx + cos(angle) * (radius * 0.2 + barHeight + 10),
        center.dy + sin(angle) * (radius * 0.2 + barHeight + 10),
      );
      
      paint.strokeWidth = 4;
      paint.strokeCap = StrokeCap.round;
      canvas.drawLine(p1, p2, paint);
      
      // Draw Grid markers
      if (i % 3 == 0) { // Every semitone (3 bins)
         final markerPaint = Paint()..color = Colors.white24..strokeWidth = 1;
         final mp1 = Offset(center.dx + cos(angle)*radius, center.dy + sin(angle)*radius);
         final mp2 = Offset(center.dx + cos(angle)*(radius+5), center.dy + sin(angle)*(radius+5));
         canvas.drawLine(mp1, mp2, markerPaint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
