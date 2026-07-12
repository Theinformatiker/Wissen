import os
import math
import pandas as pd
import folium

def haversine(lat1, lon1, lat2, lon2):
    """
    Berechnet die Entfernung zwischen zwei GPS-Punkten in Kilometern.
    """
    R = 6371.0  # Erdradius in Kilometern
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# --- PFADE ANPASSEN ---
csv_datei_pfad = './data/csv/amazfit/csv_datei1.csv'
ziel_ordner = './data/'
map_html_pfad = os.path.join(ziel_ordner, 'gps_karte.html')

try:
    # 1. Daten aus der CSV-Datei laden
    print(f"Lese Daten aus {csv_datei_pfad}...")
    df = pd.read_csv(csv_datei_pfad)
    
    # Überprüfen, ob die benötigten Spalten da sind
    if not {'lat', 'lon'}.issubset(df.columns):
        print("Fehler: Die CSV-Datei muss die Spalten 'lat' und 'lon' enthalten.")
        exit()

    lats = df['lat'].tolist()
    lons = df['lon'].tolist()

    # 2. Gesamtstrecke und Geschwindigkeiten berechnen
    gesamt_strecke = 0.0
    speeds = []
    
    # Intervall in Sekunden zwischen zwei Punkten (Amazfit zeichnet i.d.R. jede Sekunde auf)
    ZEIT_INTERVALL_SEKUNDEN = 1.0 

    for i in range(len(lats) - 1):
        # Distanz zwischen aktuellem und nächstem Punkt berechnen (in km)
        distanz = haversine(lats[i], lons[i], lats[i+1], lons[i+1])
        gesamt_strecke += distanz
        
        # Geschwindigkeit berechnen: (km / Sekunden) * 3600 = km/h
        if distanz > 0:
            geschwindigkeit_kmh = (distanz / ZEIT_INTERVALL_SEKUNDEN) * 3600.0
            # Extrem unrealistische GPS-Sprünge (z.B. > 50 km/h beim Laufen) herausfiltern
            if geschwindigkeit_kmh < 50.0:
                speeds.append(geschwindigkeit_kmh)
            else:
                speeds.append(0.0)
        else:
            speeds.append(0.0)

    # Statistiken auswerten
    if speeds:
        min_speed = min(speeds)
        max_speed = max(speeds)
        avg_speed = sum(speeds) / len(speeds)
    else:
        min_speed = max_speed = avg_speed = 0.0

    # Ergebnisse ausgeben
    print("="*40)
    print("ANALYSE AUS CSV-DATEI")
    print("="*40)
    print(f"Gesamtstrecke:                  {gesamt_strecke:.2f} km")
    print(f"Minimale Geschwindigkeit:       {min_speed:.2f} km/h")
    print(f"Maximale Geschwindigkeit:       {max_speed:.2f} km/h")
    print(f"Durchschnittsgeschwindigkeit:   {avg_speed:.2f} km/h")
    print("="*40)

    # 3. Interaktive Folium-Karte erstellen
    # Startpunkt für die Zentrierung der Karte nutzen
    start_koordinaten = [lats[0], lons[0]]
    m = folium.Map(location=start_koordinaten, zoom_start=15)

    # Koordinatenliste für die Linie (PolyLine) zusammenbauen
    coordinates = list(zip(lats, lons))
    
    # Route als Linie hinzufügen
    folium.PolyLine(coordinates, color="blue", weight=3, opacity=0.8).add_to(m)

    # Marker für Start und Ziel hinzufügen
    folium.Marker(coordinates[0], popup="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(coordinates[-1], popup="Ziel", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    # Ordner erstellen falls er fehlt und Karte speichern
    os.makedirs(ziel_ordner, exist_ok=True)
    m.save(map_html_pfad)
    print(f"Karte erfolgreich als '{map_html_pfad}' gespeichert!")

except FileNotFoundError:
    print(f"Die Datei '{csv_datei_pfad}' wurde nicht gefunden. Bitte erstelle sie zuerst.")
except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")