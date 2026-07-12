class LogEntry {
  final DateTime timestamp;
  final int bpm;
  final double lat;
  final double lon;

  LogEntry(this.timestamp, this.bpm, this.lat, this.lon);

  // Wandelt den Punkt in eine CSV-Zeile um
  String toCsvRow() => "${timestamp.toIsoformatString()}, $bpm, $lat, $lon\n";
}
