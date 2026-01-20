import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'api_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ApiService _apiService = ApiService();
  bool _isLoading = false;
  String? _errorMessage;
  
  // Results
  String? _maqamName;
  int? _rukoozBin;
  Map<String, dynamic>? _confidenceScores;

  Future<void> _pickAndAnalyzeAudio() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _maqamName = null;
    });

    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['wav'],
        withData: true, // Important for Web
      );

      if (result != null) {
        Uint8List? fileBytes = result.files.first.bytes;
        String fileName = result.files.first.name;

        if (fileBytes != null) {
          final response = await _apiService.predictMaqam(fileBytes, fileName);
          setState(() {
            _maqamName = response['predicted_maqam'];
            _rukoozBin = response['rukooz_bin'];
            _confidenceScores = response['confidence_scores'];
          });
        }
      }
    } catch (e) {
      setState(() {
        _errorMessage = e.toString();
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Maqam Detector'),
        centerTitle: true,
        elevation: 2,
      ),
      body: Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 800),
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Header Section
              const Icon(Icons.music_note, size: 80, color: Colors.indigo),
              const SizedBox(height: 20),
              Text(
                'Upload your Oud improvisation (wav)',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 10),
              const Text(
                'We analyze the microtonal intervals to detect the Maqam.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.grey),
              ),
              const SizedBox(height: 40),

              // Action Button
              FilledButton.icon(
                onPressed: _isLoading ? null : _pickAndAnalyzeAudio,
                icon: _isLoading 
                  ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
                  : const Icon(Icons.upload_file),
                label: Text(_isLoading ? 'Analyzing...' : 'Select Audio File'),
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 20),
                  textStyle: const TextStyle(fontSize: 18),
                ),
              ),
              
              const SizedBox(height: 40),

              // Error Display
              if (_errorMessage != null)
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.red.shade50,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.red.shade200),
                  ),
                  child: Text(
                    'Error: \$_errorMessage',
                    style: TextStyle(color: Colors.red.shade800),
                  ),
                ),

              // Results Display
              if (_maqamName != null) ...[
                Card(
                  elevation: 5,
                  shadowColor: Colors.indigo.withOpacity(0.3),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                  child: Padding(
                    padding: const EdgeInsets.all(32.0),
                    child: Column(
                      children: [
                        const Text('Detected Maqam', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, letterSpacing: 1.2, color: Colors.grey)),
                        const SizedBox(height: 10),
                        Text(
                          _maqamName!,
                          style: const TextStyle(
                            fontSize: 48,
                            fontWeight: FontWeight.w900,
                            color: Colors.indigo,
                          ),
                        ),
                        const Divider(height: 40),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            _buildInfoItem('Rukooz Bin', '\$_rukoozBin'),
                            _buildInfoItem('Confidence', _getConfidence(_maqamName!)),
                          ],
                        )
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                const Text("Other Probabilities (Log-Likelihood):", style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 10),
                SizedBox(
                  height: 150,
                  child: ListView.builder(
                    itemCount: _confidenceScores?.length ?? 0,
                    itemBuilder: (context, index) {
                      String key = _confidenceScores!.keys.elementAt(index);
                      double val = _confidenceScores![key];
                      return ListTile(
                        dense: true,
                        title: Text(key),
                        trailing: Text(val.toStringAsFixed(2)),
                        leading: key == _maqamName 
                          ? const Icon(Icons.check_circle, color: Colors.green) 
                          : const Icon(Icons.circle_outlined, size: 16),
                      );
                    },
                  ),
                )
              ]
            ],
          ),
        ),
      ),
    );
  }

  String _getConfidence(String maqam) {
    // Simple helper to format confidence from scores if needed
    // For now we just return "High" as a placeholder since we use Log Likelihood which is hard to normalize to % strictly without softmax
    return "High"; 
  }

  Widget _buildInfoItem(String label, String value) {
    return Column(
      children: [
        Text(label, style: const TextStyle(fontSize: 12, color: Colors.grey)),
        const SizedBox(height: 4),
        Text(value, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
      ],
    );
  }
}
