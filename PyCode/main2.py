import os
import xml.etree.ElementTree as ET
import csv
import time
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import folium

# Automatisches Erkennen des Screenshot-Modus
SCREENSHOT_MODE = None
try:
    from html2image import Html2Image
    SCREENSHOT_MODE = "html2image"
except ImportError:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        SCREENSHOT_MODE = "selenium"
    except ImportError:
        SCREENSHOT_MODE = None

# =========================================================================
# 1. STRITE DATEI-ZUORDNUNG (Paarweise Kopplung von 1 bis 6)
# =========================================================================
def find_exact_pairs():
    """
    Sucht gezielt nach den Paaren 1 bis 6 im gesamten Verzeichnisbaum,
    unabhängig davon, in welchen Unterordnern sie liegen.
    """
    gpx_ordered = []
    csv_ordered = []
    
    # Die echten Namen deiner Oberarm-Dateien (chronologisch geordnet)
    csv_names = [
        "messung_1779521039721.csv", "messung_1779523207331.csv", "messung_1780244588222.csv",
        "messung_1780246704329.csv", "messung_1780856133344.csv", "messung_1780858612298.csv"
    ]
    
    for i in range(1, 7):
        # Suche nach Zepp_X.gpx (Groß-/Kleinschreibung ignorieren via glob)
        gpx_match = glob.glob(f"**/Zepp_{i}.gpx", recursive=True) + glob.glob(f"Zepp_{i}.gpx")
        if not gpx_match:
            # Falls sie noch den langen Namen haben, als Fallback suchen
            gpx_match = glob.glob(f"**/Zepp*_{i}.gpx", recursive=True)
            
        csv_match = glob.glob(f"**/{csv_names[i-1]}", recursive=True) + glob.glob(csv_names[i-1])
        
        if gpx_match and csv_match:
            gpx_ordered.append(list(set(gpx_match))[0])
            csv_ordered.append(list(set(csv_match))[0])
        else:
            # Letzter Rettungsanker: Einfach nach Index im Ordner gehen, falls Namen verändert wurden
            all_gpx = sorted(list(set(glob.glob("**/Zepp*.gpx", recursive=True) + glob.glob("Zepp*.gpx"))))
            all_csv = sorted(list(set(glob.glob("**/messung_*.csv", recursive=True) + glob.glob("messung_*.csv"))))
            if len(all_gpx) >= 6 and len(all_csv) >= 6:
                return all_gpx[:6], all_csv[:6]
                
    return gpx_ordered, csv_ordered

# =========================================================================
# 2. GPX ZU CSV KONVERTIERUNG (Struktur angepasst an test.py)
# =========================================================================
def convert_gpx_to_csv(gpx_path, output_csv_path):
    namespaces = {
        'gpx': 'http://www.topografix.com/GPX/1/1',
        'ns3': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1'
    }
    try:
        tree = ET.parse(gpx_path)
        root = tree.getroot()
        
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'BPM', 'Latitude', 'Longitude', 'speed_native'])
            
            for trkpt in root.findall('.//gpx:trkpt', namespaces):
                lat = float(trkpt.attrib['lat'])
                lon = float(trkpt.attrib['lon'])
                
                time_el = trkpt.find('gpx:time', namespaces)
                timestamp_str = time_el.text if time_el is not None else ''
                
                if timestamp_str:
                    try:
                        utc_dt = pd.to_datetime(timestamp_str.replace('Z', '+00:00'))
                        local_dt = utc_dt.tz_convert('Europe/Berlin').tz_localize(None)
                        timestamp_str = local_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
                    except Exception:
                        pass
                
                hr_el = trkpt.find('.//ns3:hr', namespaces)
                hr = int(hr_el.text) if hr_el is not None else np.nan
                
                speed_el = trkpt.find('.//ns3:speed', namespaces)
                speed = float(speed_el.text) * 3.6 if speed_el is not None else np.nan
                
                writer.writerow([timestamp_str, hr, lat, lon, speed])
        return True
    except Exception as e:
        print(f" ❌ Fehler bei Konvertierung von {gpx_path}: {e}")
        return False

# =========================================================================
# 3. GPS-MATHEMATIK (Haversine)
# =========================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    a = np.sin(d_lat / 2.0)**2 + np.cos(p1) * np.cos(p2) * np.sin(d_lon / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def calculate_computed_speed(df, lat_col, lon_col):
    df = df.sort_values('dt').reset_index(drop=True)
    speeds = [0.0]
    for i in range(1, len(df)):
        dt_diff = (df.loc[i, 'dt'] - df.loc[i-1, 'dt']).total_seconds()
        if dt_diff > 0:
            dist = haversine_distance(df.loc[i-1, lat_col], df.loc[i-1, lon_col], 
                                      df.loc[i, lat_col], df.loc[i, lon_col])
            speeds.append((dist / dt_diff) * 3.6)
        else:
            speeds.append(0.0)
    df['computed_speed_kmh'] = speeds
    return df

# =========================================================================
# 4. MAPS & SCREENSHOTS
# =========================================================================
def generate_route_map(df_oa, df_hg, pair_label, output_dir="maps"):
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"{pair_label}_route.html")
    screenshot_path = os.path.join(output_dir, f"{pair_label}_route.png")
    
    # Karte mit CartoDB Positron initialisieren (ohne festen Zoom/Mittelpunkt)
    m = folium.Map(
        control_scale=True,
        tiles='CartoDB positron',
        attr='&copy; OpenStreetMap contributors &copy; CARTO'
    )
    
    path_oa = list(zip(df_oa['Latitude'], df_oa['Longitude']))
    path_hg = list(zip(df_hg['Latitude'], df_hg['Longitude']))
    
    # Routen-Linien zeichnen
    folium.PolyLine(path_oa, color="#1f77b4", weight=5, opacity=0.8, popup="Oberarm (CSV)").add_to(m)
    folium.PolyLine(path_hg, color="#d62728", weight=3, opacity=0.8, dash_array='5, 10', popup="Handgelenk (Zepp GPX)").add_to(m)
    
    # Start- und Ziel-Marker setzen
    folium.Marker(path_oa[0], popup="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(path_oa[-1], popup="Ziel", icon=folium.Icon(color="black", icon="stop")).add_to(m)
    
    # =========================================================================
    # AUTOMATISCHER ZOOM & ZENTRIERUNG
    # =========================================================================
    # Passt den Kartenausschnitt dynamisch an die GPS-Koordinaten an.
    # Padding sorgt für einen sauberen Sicherheitsabstand zum Bildschirmrand.
    m.fit_bounds(path_oa, padding=[50, 50])
    
    # Legende hinzufügen
    legend_html = f'''
     <div style="position: fixed; bottom: 30px; left: 30px; width: 210px; height: 75px; 
     border:2px solid grey; z-index:9999; font-size:12px; background-color:white;
     opacity: 0.9; padding: 8px; font-family: sans-serif;">
     <b>{pair_label} - Streckenvergleich</b><br>
     <span style="color:#1f77b4; font-size:16px;">▬</span> Oberarm (Lückenlos)<br>
     <span style="color:#d62728; font-size:16px;">╌╌</span> Handgelenk (Zepp Uhr)
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(html_path)
    
    # Screenshots erstellen
    if SCREENSHOT_MODE == "html2image":
        try:
            hti = Html2Image(output_path=output_dir)
            hti.screenshot(html_file=html_path, save_as=f"{pair_label}_route.png", size=(1200, 800))
        except Exception:
            pass
    elif SCREENSHOT_MODE == "selenium":
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--window-size=1200,800")
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            driver.get("file://" + os.path.abspath(html_path))
            time.sleep(2)
            driver.save_screenshot(screenshot_path)
            driver.quit()
        except Exception:
            pass
# =========================================================================
# 5. KERNLOGIK: BEREINIGUNG & PAARWEISER VERGLEICH
# =========================================================================
def analyze_data_pairs(csv_oberarm_path, csv_handgelenk_path, pair_label):
    df_oa = pd.read_csv(csv_oberarm_path)
    df_oa.columns = df_oa.columns.str.strip()
    df_hg = pd.read_csv(csv_handgelenk_path)
    df_hg.columns = df_hg.columns.str.strip()
    
    df_oa['dt'] = pd.to_datetime(df_oa['Timestamp']).dt.round('1s')
    df_hg['dt'] = pd.to_datetime(df_hg['Timestamp']).dt.round('1s')
    
    start_window = max(df_oa['dt'].min(), df_hg['dt'].min())
    end_window = min(df_oa['dt'].max(), df_hg['dt'].max())
    
    df_oa_clean = df_oa[(df_oa['dt'] >= start_window) & (df_oa['dt'] <= end_window)].drop_duplicates(subset=['dt']).copy()
    df_hg_clean = df_hg[(df_hg['dt'] >= start_window) & (df_hg['dt'] <= end_window)].drop_duplicates(subset=['dt']).copy()
    
    df_oa_clean = calculate_computed_speed(df_oa_clean, 'Latitude', 'Longitude')
    df_hg_clean = calculate_computed_speed(df_hg_clean, 'Latitude', 'Longitude')
    
    if df_hg_clean['speed_native'].isnull().all():
        df_hg_clean['speed_final'] = df_hg_clean['computed_speed_kmh']
    else:
        df_hg_clean['speed_final'] = df_hg_clean['speed_native']
        
    oa_speed = df_oa_clean[df_oa_clean['computed_speed_kmh'] < 25]['computed_speed_kmh']
    hg_speed = df_hg_clean[df_hg_clean['speed_final'] < 25]['speed_final']
    
    generate_route_map(df_oa_clean, df_hg_clean, pair_label)
    
    duration_sec = (end_window - start_window).total_seconds()
    merged = pd.merge(df_oa_clean, df_hg_clean, on='dt', how='inner', suffixes=('_oa', '_hg'))
    gps_offsets = haversine_distance(merged['Latitude_oa'], merged['Longitude_oa'], merged['Latitude_hg'], merged['Longitude_hg'])
    
    return {
        "Lauf": pair_label,
        "Dauer (Sek)": duration_sec,
        "OA_HR_Min": df_oa_clean['BPM'].min(), "OA_HR_Max": df_oa_clean['BPM'].max(), "OA_HR_Avg": round(df_oa_clean['BPM'].mean(), 1),
        "HG_HR_Min": df_hg_clean['BPM'].min(), "HG_HR_Max": df_hg_clean['BPM'].max(), "HG_HR_Avg": round(df_hg_clean['BPM'].mean(), 1),
        "OA_HR_Loss(s)": df_oa_clean['BPM'].isna().sum(), "HG_HR_Loss(s)": df_hg_clean['BPM'].isna().sum(),
        "OA_Speed_Avg": round(oa_speed.mean(), 1), "HG_Speed_Avg": round(hg_speed.mean(), 1),
        "GPS_Abw_Ø(m)": round(gps_offsets.mean(), 2), "GPS_Abw_Max(m)": round(gps_offsets.max(), 2)
    }

# =========================================================================
# 6. ENGINE PIPELINE
# =========================================================================
if __name__ == "__main__":
    print("=== PIPELINE GESTARTET ===")
    
    gpx_files, csv_oberarm_files = find_exact_pairs()
    
    if len(gpx_files) < 6 or len(csv_oberarm_files) < 6:
        print(f"❌ Fehler: Es müssen mindestens 6 GPX und 6 CSV Dateien im Projekt existieren.")
        print(f"Gefunden: {len(gpx_files)} GPX, {len(csv_oberarm_files)} CSV.")
        exit()
        
    print(f"-> {len(gpx_files)} GPX-Dateien und {len(csv_oberarm_files)} CSV-Dateien erfolgreich gepaart.")
    
    final_reports = []
    os.makedirs("zepp_csv_output", exist_ok=True)
    
    for i in range(6):
        gpx_source = gpx_files[i]
        csv_source = csv_oberarm_files[i]
        
        # Generiert permanent den Ordner und die Dateien Zepp_1.csv bis Zepp_6.csv
        zepp_csv_target = f"zepp_csv_output/Zepp_{i+1}.csv"
        success = convert_gpx_to_csv(gpx_source, zepp_csv_target)
        
        if success:
            print(f"[Lauf {i+1}] Konvertiert: {os.path.basename(gpx_source)} -> Zepp_{i+1}.csv")
            rep = analyze_data_pairs(csv_source, zepp_csv_target, f"Lauf_{i+1}")
            final_reports.append(rep)
            
    if final_reports:
        df_results = pd.DataFrame(final_reports)
        print("\n" + "="*95)
        print("FINALE BEREINIGTE SYNC-VERGLEICHSANALYSE (Lauf 1 bis 6)")
        print("="*95)
        print(df_results.to_string(index=False))
        
        # Sicherer CSV-Export mit Schutz gegen Excel-Sperren (PermissionError)
        export_filename = "trainings_vergleichs_bericht.csv"
        try:
            df_results.to_csv(export_filename, index=False, sep=";")
            print(f"\n[Erfolg] Bericht unter '{export_filename}' gesichert.")
        except PermissionError:
            timestamp_suffix = datetime.now().strftime("%H%M%S")
            fallback_filename = f"trainings_vergleichs_bericht_{timestamp_suffix}.csv"
            df_results.to_csv(fallback_filename, index=False, sep=";")
            print(f"\n⚠️ Hinweis: '{export_filename}' war in Excel geöffnet!")
            print(f"-> Bericht wurde stattdessen als '{fallback_filename}' gespeichert. Bitte Excel schließen.")
            
        print("Interaktive HTML-Karten und Screenshots liegen im Ordner '/maps'.")
        print("Die umgewandelten Zepp-CSVs (Zepp_1.csv bis Zepp_6.csv) liegen im Ordner '/zepp_csv_output'.")