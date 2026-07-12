import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'package:geolocator/geolocator.dart';

void main() {
  FlutterBluePlus.setLogLevel(LogLevel.info, color: true);
  runApp(const HeartRateApp());
}

// 1. Unsere Daten-Klasse (definiert eine einzelne Zeile in der CSV)
class LogEntry {
  final DateTime timestamp;
  final int bpm;
  final double lat;
  final double lon;

  LogEntry(this.timestamp, this.bpm, this.lat, this.lon);

  String toCsvRow() => "${timestamp.toIso8601String()}, $bpm, $lat, $lon\n";
}

class HeartRateApp extends StatelessWidget {
  const HeartRateApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Wissenschaftliches Arbeiten - Tracker',
      theme: ThemeData.dark().copyWith(
        primaryColor: Colors.blue[900],
        scaffoldBackgroundColor: const Color(0xFF1E1E1E),
      ),
      home: const SensorScreen(),
    );
  }
}

class SensorScreen extends StatefulWidget {
  const SensorScreen({Key? key}) : super(key: key);

  @override
  State<SensorScreen> createState() => _SensorScreenState();
}

class _SensorScreenState extends State<SensorScreen> {
  String statusText = "Bereit zum Scannen";
  String currentBpm = "---";
  bool isScanning = false;

  // NEU: Status-Variable für die Aufnahme
  bool isRecording = false;

  BluetoothDevice? hw9Device;
  StreamSubscription<List<ScanResult>>? scanSubscription;

  // 2. Unser Daten-Buffer
  List<LogEntry> _dataBuffer = [];

  @override
  void initState() {
    super.initState();
    _requestPermissions();
  }

  Future<void> _requestPermissions() async {
    await [
      Permission.location,
      Permission.bluetoothScan,
      Permission.bluetoothConnect,
    ].request();
  }

  Future<void> startScan() async {
    setState(() {
      isScanning = true;
      statusText = "Suche nach HW9...";
      currentBpm = "---";
      isRecording = false; // Aufnahme beim Neustart sicherheitshalber stoppen
      _dataBuffer.clear(); // Buffer beim Neustart leeren
    });

    try {
      await FlutterBluePlus.startScan(timeout: const Duration(seconds: 15));

      scanSubscription = FlutterBluePlus.scanResults.listen((results) async {
        for (ScanResult r in results) {
          if (r.device.advName.contains("HW9") ||
              r.device.platformName.contains("HW9")) {
            await FlutterBluePlus.stopScan();
            setState(() {
              statusText = "HW9 gefunden! Verbinde...";
              hw9Device = r.device;
            });
            _connectToDevice(r.device);
            break;
          }
        }
      });
    } catch (e) {
      setState(() {
        statusText = "Scan Fehler: $e";
        isScanning = false;
      });
    }
  }

  Future<void> _connectToDevice(BluetoothDevice device) async {
    try {
      await device.connect(autoConnect: false, license: License.free);
      setState(() {
        statusText = "Verbunden! Warte auf Start...";
        isScanning = false;
      });

      List<BluetoothService> services = await device.discoverServices();
      bool heartRateFound = false;

      for (BluetoothService service in services) {
        for (BluetoothCharacteristic characteristic
            in service.characteristics) {
          if (characteristic.uuid.toString().toLowerCase().contains("2a37")) {
            heartRateFound = true;
            await characteristic.setNotifyValue(true);

            characteristic.onValueReceived.listen((value) async {
              if (value.isNotEmpty) {
                int flag = value[0];
                int format = flag & 0x01;
                int bpm = 0;

                if (format == 0 && value.length > 1) {
                  bpm = value[1];
                } else if (format == 1 && value.length > 2) {
                  bpm = value[1] + (value[2] << 8);
                }

                if (bpm > 0) {
                  // UI aktualisieren, damit man den Puls immer sieht
                  setState(() {
                    currentBpm = bpm.toString();
                  });

                  // NEU: Nur in den Buffer schreiben, wenn "Start" gedrückt wurde
                  if (isRecording) {
                    try {
                      Position position = await Geolocator.getCurrentPosition(
                        desiredAccuracy: LocationAccuracy.high,
                      );

                      setState(() {
                        statusText =
                            "Logge Daten... (${_dataBuffer.length + 1} Punkte)";
                        _dataBuffer.add(
                          LogEntry(
                            DateTime.now(),
                            bpm,
                            position.latitude,
                            position.longitude,
                          ),
                        );
                      });
                    } catch (e) {
                      setState(() {
                        statusText =
                            "Logge Daten (Kein GPS)... (${_dataBuffer.length + 1} Punkte)";
                        _dataBuffer.add(
                          LogEntry(DateTime.now(), bpm, 0.0, 0.0),
                        );
                      });
                    }
                  }
                }
              }
            });
          }
        }
      }

      if (!heartRateFound) {
        setState(() {
          statusText = "Kein Puls-Sensor (2a37) gefunden.";
        });
      }
    } catch (e) {
      setState(() {
        statusText = "Verbindungsfehler: $e";
        isScanning = false;
      });
    }
  }

  // NEU: Funktion zum Starten/Stoppen der Aufnahme
  void _toggleRecording() {
    setState(() {
      if (isRecording) {
        isRecording = false;
        statusText = "Aufnahme gestoppt. Bereit zum Speichern.";
      } else {
        isRecording = true;
        statusText = "Aufnahme läuft...";
      }
    });
  }

  // 4. Die Speicher-Funktion
  Future<void> _saveDataToFile() async {
    if (_dataBuffer.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            "Keine Daten zum Speichern vorhanden!",
            style: TextStyle(color: Colors.white),
          ),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    try {
      final directory = await getExternalStorageDirectory();
      final file = File(
        '${directory!.path}/messung_${DateTime.now().millisecondsSinceEpoch}.csv',
      );

      String csvContent = "Timestamp, BPM, Latitude, Longitude\n";
      for (var entry in _dataBuffer) {
        csvContent += entry.toCsvRow();
      }

      await file.writeAsString(csvContent);

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            "Erfolg! Gespeichert in: ${file.path}",
            style: const TextStyle(color: Colors.white),
          ),
          backgroundColor: Colors.green,
          duration: const Duration(seconds: 5),
        ),
      );

      // Buffer leeren und Aufnahme stoppen nach dem Speichern
      setState(() {
        isRecording = false;
        _dataBuffer.clear();
        statusText = "Daten gesichert.";
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            "Fehler beim Speichern: $e",
            style: const TextStyle(color: Colors.white),
          ),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Future<void> _disconnect() async {
    if (hw9Device != null) {
      await hw9Device!.disconnect();
      setState(() {
        hw9Device = null;
        isRecording = false; // Aufnahme beim Trennen stoppen
        currentBpm = "---";
        statusText = "Getrennt.";
      });
    }
  }

  @override
  void dispose() {
    scanSubscription?.cancel();
    _disconnect();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        toolbarHeight: 20,
        title: const Text(
          "Vital Tracker (Flutter)",
          style: TextStyle(fontSize: 12, color: Colors.white70),
        ),
        backgroundColor: Colors.blue[900],
        centerTitle: true,
      ),
      body: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.favorite, color: Colors.redAccent, size: 20),
              const SizedBox(height: 5),
              const Text(
                "Herzfrequenz",
                style: TextStyle(fontSize: 12, color: Colors.grey),
              ),
              Text(
                currentBpm,
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              const Text(
                "BPM",
                style: TextStyle(fontSize: 20, color: Colors.redAccent),
              ),
              const SizedBox(height: 5),
              Text(
                statusText,
                style: const TextStyle(fontSize: 12, color: Colors.grey),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 5),

              // Connect / Disconnect Button
              ElevatedButton.icon(
                onPressed: isScanning
                    ? null
                    : (hw9Device == null ? startScan : _disconnect),
                icon: Icon(
                  hw9Device == null
                      ? Icons.bluetooth_searching
                      : Icons.bluetooth_disabled,
                ),
                label: Text(
                  hw9Device == null ? "HW9 Verbinden" : "Trennen",
                  style: const TextStyle(fontSize: 12),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: hw9Device == null
                      ? Colors.blue[700]
                      : Colors.red[800],
                  padding: const EdgeInsets.symmetric(
                    horizontal: 30,
                    vertical: 5,
                  ),
                ),
              ),

              const SizedBox(height: 5),

              // NEU: Start / Stop Button für die Messung
              ElevatedButton.icon(
                // Button ist nur klickbar, wenn das Gerät verbunden ist
                onPressed: hw9Device != null ? _toggleRecording : null,
                icon: Icon(isRecording ? Icons.pause : Icons.play_arrow),
                label: Text(
                  isRecording ? "Messung Pausieren" : "Messung Starten",
                  style: const TextStyle(fontSize: 12),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: isRecording
                      ? Colors.orange[700]
                      : Colors.green[600],
                  padding: const EdgeInsets.symmetric(
                    horizontal: 30,
                    vertical: 5,
                  ),
                ),
              ),

              const SizedBox(height: 7),

              // Session Speichern Button
              ElevatedButton.icon(
                onPressed: _dataBuffer.isEmpty ? null : _saveDataToFile,
                icon: const Icon(Icons.save_alt),
                label: const Text(
                  "Session Speichern",
                  style: TextStyle(fontSize: 12),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.teal[700],
                  padding: const EdgeInsets.symmetric(
                    horizontal: 30,
                    vertical: 5,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
