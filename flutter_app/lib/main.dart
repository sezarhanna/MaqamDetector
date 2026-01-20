import 'package:flutter/material.dart';
import 'screens/dashboard.dart';

void main() {
  runApp(const MaqamApp());
}

class MaqamApp extends StatelessWidget {
  const MaqamApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Maqam Detector 2.0',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF050505), // Obsidian Black
        primaryColor: const Color(0xFF00E5FF), // Cyan Accent
        fontFamily: 'RobotoMono', // Monospace for 'Research' feel
        cardColor: const Color(0xFF1E1E1E),
        useMaterial3: true,
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00E5FF),
          secondary: Color(0xFFFF0055), // Neon Red
          surface: Color(0xFF121212),
        ),
      ),
      home: const DashboardScreen(),
    );
  }
}
