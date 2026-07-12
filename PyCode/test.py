import xml.etree.ElementTree as ET
import csv
import os
from datetime import datetime
import zoneinfo # Ab Python 3.9 integriert. Für ältere Versionen: von dateutil.parser import parse

# Pfade zu deinen Dateien (relativ zum Skript)
gpx_datei_pfad = './data/Zepp20260523085643.gpx'
csv_datei_pfad = './data/csv/amazfit/csv_datei1.csv'

# Namespaces definieren, die in der GPX-Datei genutzt werden
namespaces = {
    'gpx': 'http://www.topografix.com/GPX/1/1',
    'ns3': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1'
}

try:
    # Verzeichnis automatisch erstellen, falls es noch nicht existiert
    os.makedirs(os.path.dirname(csv_datei_pfad), exist_ok=True)

    # GPX-Datei laden und Struktur parsen
    tree = ET.parse(gpx_datei_pfad)
    root = tree.getroot()

    # CSV-Datei zum Schreiben öffnen
    with open(csv_datei_pfad, mode='w', newline='', encoding='utf-8') as csv_datei:
        writer = csv.writer(csv_datei)
        
        # Spaltenüberschriften schreiben
        writer.writerow(['Timestamp', 'BPM', 'Latitude', 'Longitude'])
        
        # Alle Trackpunkte (<trkpt>) durchlaufen
        for trkpt in root.findall('.//gpx:trkpt', namespaces):
            # Koordinaten auslesen
            lat = trkpt.get('lat')
            lon = trkpt.get('lon')
            
            # Zeitstempel auslesen (<time>-Tag innerhalb des Trackpoints)
            time_element = trkpt.find('gpx:time', namespaces)
            timestamp = time_element.text if time_element is not None else ''
            
            # --- NEU: Zeitstempel von UTC in lokale Berliner Zeit umrechnen ---
            if timestamp:
                try:
                    # Entfernt das 'Z' und parst den String in ein datetime-Objekt (UTC)
                    utc_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    # In die Zeitzone von Berlin umrechnen (berücksichtigt Sommer-/Winterzeit automatisch)
                    local_time = utc_time.astimezone(zoneinfo.ZoneInfo("Europe/Berlin"))
                    # Formatieren ohne Zeitzonen-Suffix (wie bei deiner Flutter-App)
                    timestamp = local_time.strftime('%Y-%m-%dT%H:%M:%S.%f')
                except Exception:
                    pass # Falls ein Zeitstempel mal fehlerhaft ist, bleibt er unverändert
            # -----------------------------------------------------------------
            
            # Herzfrequenz auslesen (<ns3:hr> innerhalb der Extensions)
            hr_element = trkpt.find('.//ns3:hr', namespaces)
            bpm = hr_element.text if hr_element is not None else ''
            
            # Zeile schreiben
            writer.writerow([timestamp, bpm, lat, lon])

    print(f"Erfolgreich extrahiert! Die Datei wurde unter '{csv_datei_pfad}' gespeichert.")

except FileNotFoundError:
    print(f"Die Datei '{gpx_datei_pfad}' wurde nicht gefunden. Bitte überprüfe den Pfad.")
except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")