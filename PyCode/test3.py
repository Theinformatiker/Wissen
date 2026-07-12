import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. DATEN EINLESEN UND BEREINIGEN
# -------------------------------------------------------------------------
# Lade die beiden CSV-Dateien
df1 = pd.read_csv('data/csv/flutter/messung_1779521039721.csv')
df2 = pd.read_csv('data/csv/amazfit/csv_datei2.csv')

# Entferne eventuelle Leerzeichen aus den Spaltennamen (z.B. ' BPM' -> 'BPM')
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Konvertiere die Zeitstempel in echte Datetime-Objekte
df1['dt'] = pd.to_datetime(df1['Timestamp'])
df2['dt'] = pd.to_datetime(df2['Timestamp'])

# Sortiere die Daten chronologisch
df1 = df1.sort_values('dt').reset_index(drop=True)
df2 = df2.sort_values('dt').reset_index(drop=True)


# 2. ZEITANALYSE: SAMPLING-INTERVALLE (ABTASTRATE)
# -------------------------------------------------------------------------
# Berechne den Zeitabstand zwischen aufeinanderfolgenden Punkten in Sekunden
df1['time_diff'] = df1['dt'].diff().dt.total_seconds()
df2['time_diff'] = df2['dt'].diff().dt.total_seconds()

plt.figure(figsize=(10, 5))
plt.hist(df1['time_diff'].dropna(), bins=np.arange(0, 5, 0.25), alpha=0.5, label='Gerät 1 (messung...)', color='blue', density=True)
plt.hist(df2['time_diff'].dropna(), bins=np.arange(0, 5, 0.25), alpha=0.5, label='Gerät 2 (csv_datei2)', color='orange', density=True)
plt.title('Verteilung der Messintervalle (Abtastrate in Sekunden)')
plt.xlabel('Zeitabstand zwischen Datenpunkten (Sekunden)')
plt.ylabel('Relative Häufigkeit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('zeit_intervalle_histogramm.png')
plt.close()


# 3. MATHEMATISCHE BERECHNUNG VON DISTANZ UND GESCHWINDIGKEIT
# -------------------------------------------------------------------------
def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Berechnet die Distanz zwischen zwei GPS-Punkten in Metern 
    mithilfe der Haversine-Formel (Berücksichtigung der Erdkrümmung).
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_meters = 6371000
    return c * earth_radius_meters

# --- Berechnungen für Gerät 1 ---
df1['dist_delta'] = haversine_distance(df1['Longitude'].shift(), df1['Latitude'].shift(), df1['Longitude'], df1['Latitude']).fillna(0)
df1['cum_dist_km'] = df1['dist_delta'].cumsum() / 1000.0
df1['speed_kmh'] = (df1['dist_delta'] / df1['time_diff']) * 3.6

# --- Berechnungen für Gerät 2 ---
df2['dist_delta'] = haversine_distance(df2['Longitude'].shift(), df2['Latitude'].shift(), df2['Longitude'], df2['Latitude']).fillna(0)
df2['cum_dist_km'] = df2['dist_delta'].cumsum() / 1000.0
df2['speed_kmh'] = (df2['dist_delta'] / df2['time_diff']) * 3.6


# 4. PLOT: KUMULIERTE DISTANZ ÜBER DIE ZEIT
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df1['dt'], df1['cum_dist_km'], label='Gerät 1', color='blue', linewidth=2)
plt.plot(df2['dt'], df2['cum_dist_km'], label='Gerät 2', color='orange', linestyle='--', linewidth=2)
plt.title('Zurückgelegte Distanz über die Zeit')
plt.xlabel('Uhrzeit')
plt.ylabel('Distanz (km)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('kumulierte_distanz.png')
plt.close()


# 5. PLOT: GESCHWINDIGKEITSPROFIL (GEGLÄTTET)
# -------------------------------------------------------------------------
# Da rohe GPS-Geschwindigkeiten stark springen, nutzen wir ein gleitendes Fenster (z.B. 15 Sekunden)
df1['speed_kmh_smooth'] = df1['speed_kmh'].rolling(window=15, min_periods=1).mean()
df2['speed_kmh_smooth'] = df2['speed_kmh'].rolling(window=15, min_periods=1).mean()

plt.figure(figsize=(10, 5))
plt.plot(df1['dt'], df1['speed_kmh_smooth'], label='Gerät 1 (15s geglättet)', color='blue', alpha=0.8)
plt.plot(df2['dt'], df2['speed_kmh_smooth'], label='Gerät 2 (15s geglättet)', color='orange', alpha=0.8, linestyle='--')
plt.title('Geschwindigkeitsprofil über die Zeit')
plt.xlabel('Uhrzeit')
plt.ylabel('Geschwindigkeit (km/h)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('geschwindigkeitsprofil.png')
plt.close()


# 6. PLOT: HERZFREQUENZ-ZONEN VERTEILUNG
# -------------------------------------------------------------------------
# Definition klassischer Belastungszonen (kannst du an deine HF-max anpassen)
bins = [0, 120, 140, 160, 180, 220]
labels = ['<120 (Regeneration)', '120-140 (GA1 / Fettverbrennung)', '140-160 (GA2 / Ausdauer)', '160-180 (EB / Entwicklung)', '>180 (Spitzenbereich)']

# Wir nutzen Gerät 1, da es lückenlose Pulswerte besitzt
df1['hr_zone'] = pd.cut(df1['BPM'], bins=bins, labels=labels)
# Prozentualen Anteil berechnen
zone_counts = df1['hr_zone'].value_counts(normalize=True).reindex(labels) * 100

plt.figure(figsize=(9, 5))
zone_counts.plot(kind='bar', color='teal', alpha=0.8)
plt.title('Verteilung der Herzfrequenz-Zonen (Gerät 1)')
plt.xlabel('Herzfrequenz-Zone (BPM)')
plt.ylabel('Anteil an der Gesamtlaufzeit (%)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig('herzfrequenz_zonen.png')
plt.close()


# 7. KONSOLEN-AUSGABE DER STATISTIKEN
# -------------------------------------------------------------------------
print("=== ZUSAMMENFASSUNG DER ANALYSE ===")
print(f"Gerät 1 - Start: {df1['dt'].iloc[0]} | Ende: {df1['dt'].iloc[-1]}")
print(f"Gerät 2 - Start: {df2['dt'].iloc[0]} | Ende: {df2['dt'].iloc[-1]}")
print(f"Gesamtdistanz Gerät 1: {df1['cum_dist_km'].max():.2f} km")
print(f"Gesamtdistanz Gerät 2: {df2['cum_dist_km'].max():.2f} km")
print(f"Durchschnittlicher Puls Gerät 1: {df1['BPM'].mean():.1f} BPM")
print(f"Durchschnittlicher Puls Gerät 2 (vorhandene Werte): {df2['BPM'].dropna().mean():.1f} BPM")