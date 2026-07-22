import os
import xml.etree.ElementTree as ET
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr  # For calculating scientific correlation

# =========================================================================
# 1. FILE MAPPING
# =========================================================================
def find_exact_pairs(zepp_dir=os.path.join("data", "csv", "amazfit"),
                      messung_dir=os.path.join("data", "csv", "flutter")):
    """
    Sucht Zepp_i.csv in zepp_dir und Messung_i.csv in messung_dir.
    Passend zur Projektstruktur:
        data/csv/amazfit/Zepp_1..6.csv
        data/csv/flutter/Messung_1..6.csv
    Keine GPX-Konvertierung noetig, da beide Geraete bereits als CSV
    vorliegen (Timestamp, BPM, Latitude, Longitude, speed_native).
    """
    zepp_ordered = [os.path.join(zepp_dir, f"Zepp_{i}.csv") for i in range(1, 7)]
    csv_ordered = [os.path.join(messung_dir, f"Messung_{i}.csv") for i in range(1, 7)]
    missing = [p for p in zepp_ordered + csv_ordered if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Folgende Dateien wurden nicht gefunden:\n  " + "\n  ".join(missing) +
            f"\nErwartet werden Zepp_1..6.csv in '{zepp_dir}' und Messung_1..6.csv in "
            f"'{messung_dir}' (relativ zum Arbeitsverzeichnis, aus dem vk.py gestartet wird)."
        )
    return zepp_ordered, csv_ordered

# =========================================================================
# 2. GPX CONVERSION
# =========================================================================
def convert_gpx_to_csv(gpx_path, output_csv_path):
    namespaces = {'gpx': 'http://www.topografix.com/GPX/1/1', 'ns3': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1'}
    try:
        tree = ET.parse(gpx_path)
        root = tree.getroot()
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'BPM', 'Latitude', 'Longitude', 'speed_native'])
            for trkpt in root.findall('.//gpx:trkpt', namespaces):
                lat = float(trkpt.attrib['lat']); lon = float(trkpt.attrib['lon'])
                time_el = trkpt.find('gpx:time', namespaces)
                ts = pd.to_datetime(time_el.text.replace('Z', '+00:00')).tz_convert('Europe/Berlin').tz_localize(None).strftime('%Y-%m-%dT%H:%M:%S.%f') if time_el is not None else ''
                hr = int(trkpt.find('.//ns3:hr', namespaces).text) if trkpt.find('.//ns3:hr', namespaces) is not None else np.nan
                speed = float(trkpt.find('.//ns3:speed', namespaces).text) * 3.6 if trkpt.find('.//ns3:speed', namespaces) is not None else np.nan
                writer.writerow([ts, hr, lat, lon, speed])
        return True
    except: return False

# =========================================================================
# 3. GPS & PLOTTING FUNCTIONS
# =========================================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def calculate_gps_offset(df_arm, df_wrist):
    """
    Berechnet den GPS-Offset (Distanz zwischen Oberarm- und Handgelenk-Position)
    fuer jeden synchronisierten Zeitpunkt einer Session.
    Gibt (mean_offset_m, max_offset_m, offset_series) zurueck.
    Wichtig: das ist die Grundlage, um NICHT nur einen einzelnen Lauf (Trial 3)
    als repraesentativen Wert zu zitieren, sondern den Durchschnitt ueber
    alle N Laeufe korrekt zu berichten (siehe Gutachten-Punkt 4).
    """
    merged = pd.merge(df_arm, df_wrist, on='dt', how='inner', suffixes=('_arm', '_wrist')).dropna(
        subset=['Latitude_arm', 'Longitude_arm', 'Latitude_wrist', 'Longitude_wrist'])
    if merged.empty:
        return np.nan, np.nan, pd.Series(dtype=float)
    offset = haversine(merged['Latitude_arm'], merged['Longitude_arm'],
                        merged['Latitude_wrist'], merged['Longitude_wrist'])
    offset.index = merged['dt']
    return offset.mean(), offset.max(), offset

def calculate_cross_correlation_lag(df_arm, df_wrist, max_lag_s=60):
    """
    Bestimmt per Kreuzkorrelation den zeitlichen Versatz (Lag) zwischen der
    Oberarm- und der Handgelenk-HF-Zeitreihe -- das quantifiziert, was im
    Paper bisher nur visuell als "smoothing lag" beschrieben wird (Section 5).

    Beide Serien werden auf ein gemeinsames 1-Sekunden-Raster gebracht und
    linear interpoliert (so bleibt die Zeitreihe auch ueber die Dropout-
    Luecken des Handgelenksensors hinweg durchgehend). Anschliessend wird
    die Handgelenk-Serie um lag in {-max_lag_s, ..., +max_lag_s} Sekunden
    verschoben und jeweils mit der Oberarm-Serie korreliert.

    Konvention: positiver Lag = die Handgelenk-Kurve hinkt der Oberarm-
    Kurve um so viele Sekunden hinterher (Handgelenk reagiert spaeter).

    Gibt (best_lag_s, best_corr, corr_series) zurueck, wobei corr_series
    ueber den Lag indiziert ist (fuer den Lag-Kurven-Plot).
    """
    if df_arm.empty or df_wrist.empty:
        return np.nan, np.nan, pd.Series(dtype=float)

    start = max(df_arm['dt'].min(), df_wrist['dt'].min())
    end = min(df_arm['dt'].max(), df_wrist['dt'].max())
    if pd.isna(start) or pd.isna(end) or start >= end:
        return np.nan, np.nan, pd.Series(dtype=float)

    idx = pd.date_range(start, end, freq='1s')
    arm_s = (df_arm.drop_duplicates('dt').set_index('dt')['BPM']
             .reindex(idx).interpolate('time'))
    wrist_s = (df_wrist.drop_duplicates('dt').set_index('dt')['BPM']
               .reindex(idx).interpolate('time'))

    corrs = {}
    for lag in range(-max_lag_s, max_lag_s + 1):
        # shift(-lag): wrist_shifted[t] = wrist_s[t+lag]; wenn das Handgelenk
        # tatsaechlich um `lag` Sekunden hinterherhinkt, entspricht das dann
        # wieder dem Oberarmwert zum Zeitpunkt t -> Korrelation maximal bei
        # positivem lag, wie in der Docstring-Konvention beschrieben.
        wrist_shifted = wrist_s.shift(-lag)
        valid = arm_s.notna() & wrist_shifted.notna()
        if valid.sum() < 10:
            continue
        corrs[lag] = arm_s[valid].corr(wrist_shifted[valid])

    corr_series = pd.Series(corrs)
    if corr_series.empty or corr_series.isna().all():
        return np.nan, np.nan, corr_series

    best_lag = corr_series.idxmax()
    if abs(best_lag) >= max_lag_s - 1:
        print(f"    WARNING: best lag ({best_lag:+d}s) is at/near the search "
              f"boundary (+/-{max_lag_s}s) -- the true optimum may lie outside "
              f"the tested range and this estimate should be treated as unreliable.")
    return best_lag, corr_series[best_lag], corr_series


def calculate_computed_speed(df):
    df = df.sort_values('dt').reset_index(drop=True)
    speeds = [0.0]
    for i in range(1, len(df)):
        dt = (df.loc[i, 'dt'] - df.loc[i-1, 'dt']).total_seconds()
        if dt > 0: speeds.append((haversine(df.loc[i-1, 'Latitude'], df.loc[i-1, 'Longitude'], df.loc[i, 'Latitude'], df.loc[i, 'Longitude']) / dt) * 3.6)
        else: speeds.append(0.0)
    df['computed_speed_kmh'] = speeds
    return df

def calculate_total_distance_km(df):
    """
    Summiert die Haversine-Distanz zwischen aufeinanderfolgenden GPS-Punkten
    einer Session auf (chronologisch sortiert). Gibt die Gesamtdistanz in km
    zurueck. Wird sowohl fuer OA (Upper Arm) als auch HG (Wrist) separat
    aufgerufen -- daraus ergibt sich die Distanz-Diskrepanz zwischen den
    Geraeten (Table 1: "OA/HG Distance", bisher ohne Fehlermetrik).
    """
    d = df.sort_values('dt').dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
    if len(d) < 2:
        return np.nan
    dist_m = haversine(d['Latitude'].values[:-1], d['Longitude'].values[:-1],
                        d['Latitude'].values[1:], d['Longitude'].values[1:])
    return float(np.sum(dist_m)) / 1000.0

def calculate_distance_error(oa_km, hg_km):
    """
    Relativer Distanz-Fehler zwischen Oberarm- (Referenz) und
    Handgelenk-Distanz, in Prozent. |OA - HG| / OA * 100.
    """
    if oa_km is None or np.isnan(oa_km) or oa_km == 0:
        return np.nan
    return abs(oa_km - hg_km) / oa_km * 100.0

def calculate_speed_agreement(df_arm, df_wrist):
    """
    Analog zu den HR-Agreement-Metriken (Pearson r, MAE, Bias): vergleicht
    die geglaettete Oberarm-Geschwindigkeit (5s rolling mean, wie im
    Speed-Profile-Plot verwendet) mit der rohen Handgelenk-Geschwindigkeit
    zu jedem gemeinsamen Zeitstempel. Bisher wurde Speed nur geplottet,
    nie quantifiziert -- das schliesst diese Luecke.

    Gibt (pearson_r, mae_kmh, bias_kmh) zurueck. Bias = Upper Arm - Wrist.
    """
    arm = df_arm[['dt', 'speed_native']].copy()
    arm['speed_smoothed'] = arm['speed_native'].rolling(window=5, center=True, min_periods=1).mean()
    wrist = df_wrist[['dt', 'speed_native']].copy()

    merged = pd.merge(arm, wrist, on='dt', how='inner', suffixes=('_arm', '_wrist')).dropna(
        subset=['speed_smoothed', 'speed_native_wrist'])
    if merged.empty or len(merged) < 10:
        return np.nan, np.nan, np.nan

    speed_arm = merged['speed_smoothed']
    speed_wrist = merged['speed_native_wrist']
    r, _ = pearsonr(speed_arm, speed_wrist)
    mae = float(np.mean(np.abs(speed_arm - speed_wrist)))
    bias = float((speed_arm - speed_wrist).mean())
    return float(r), mae, bias

def generate_all_plots(df_arm, df_wrist, pair_label, date_str):
    run_folder = os.path.join("plots", pair_label)
    os.makedirs(run_folder, exist_ok=True)
    
    main_title = f"{pair_label} - {date_str}"
    
    # Common helper function for axis formatting (time series)
    def format_x_axis(ax, y_label):
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1)) 
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.set_xlabel('Time (hh:mm)', fontsize=10, labelpad=10)
        ax.set_ylabel(y_label, fontsize=10, labelpad=10)

    # ---------------------------------------------------------------------
    # PLOT 1: Heart Rate Trend
    # ---------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df_arm['dt'], df_arm['BPM'], label='Upper Arm (Sensor)', alpha=0.7, linewidth=1.5)
    plt.plot(df_wrist['dt'], df_wrist['BPM'], label='Wrist (Smartwatch)', alpha=0.7, linewidth=1.5)
    
    plt.title(f'{main_title}: Heart Rate Trend', fontsize=12, fontweight='bold', pad=15)
    format_x_axis(plt.gca(), 'Heart Rate (BPM)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, f"{pair_label}_HR_Trend.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------------------
    # PLOT 2: Speed Profile
    # ---------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    arm_smoothed = df_arm['speed_native'].rolling(window=5, center=True, min_periods=1).mean()
    
    plt.plot(df_arm['dt'], arm_smoothed, label='Upper Arm (GPS computed / smoothed)', alpha=0.7, linewidth=1.5)
    plt.plot(df_wrist['dt'], df_wrist['speed_native'], label='Wrist (Sensor)', alpha=0.7, linewidth=1.5)
    
    plt.title(f'{main_title}: Speed Profile', fontsize=12, fontweight='bold', pad=15)
    format_x_axis(plt.gca(), 'Speed (km/h)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, f"{pair_label}_Speed_Profile.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------------------
    # PREPARATION FOR METHOD COMPARISONS (Merging via timestamp)
    # ---------------------------------------------------------------------
    merged = pd.merge(df_arm, df_wrist, on='dt', how='inner', suffixes=('_arm', '_wrist')).dropna(subset=['BPM_arm', 'BPM_wrist'])

    # Initialisierung, damit die Werte auch dann definiert sind (als NaN),
    # wenn 'merged' leer ist -- verhindert NameError beim Rueckgabe-Tupel.
    corr_coeff, mae, bias, sd, upper_loa, lower_loa = (np.nan,) * 6

    if not merged.empty:
        bpm_arm = merged['BPM_arm']
        bpm_wrist = merged['BPM_wrist']
        
        # Calculation of scientific metrics
        corr_coeff, _ = pearsonr(bpm_arm, bpm_wrist)
        mae = np.mean(np.abs(bpm_arm - bpm_wrist))

        # ---------------------------------------------------------------------
        # PLOT 3: Scatterplot (Correlation Analysis)
        # ---------------------------------------------------------------------
        plt.figure(figsize=(7, 7))
        plt.scatter(bpm_arm, bpm_wrist, alpha=0.4, color='#6a0dad', edgecolors='none', s=20)
        
        # Identity line (y = x)
        max_val = max(bpm_arm.max(), bpm_wrist.max())
        min_val = min(bpm_arm.min(), bpm_wrist.min())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.5, label='Perfect Agreement (y=x)')
        
        plt.title(f'{main_title}: HR Correlation\n(Pearson r = {corr_coeff:.2f} | MAE = {mae:.1f} BPM)', fontsize=11, fontweight='bold', pad=12)
        plt.xlabel('Reference: Upper Arm Sensor (BPM)', fontsize=10)
        plt.ylabel('Test Device: Wrist Smartwatch (BPM)', fontsize=10)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(run_folder, f"{pair_label}_HR_Scatterplot.png"), dpi=300)
        plt.close()

        # ---------------------------------------------------------------------
        # PLOT 4: Bland-Altman Plot (Systematic Deviations)
        # ---------------------------------------------------------------------
        mean_bpm = (bpm_arm + bpm_wrist) / 2
        diff_bpm = bpm_arm - bpm_wrist
        bias = diff_bpm.mean()                   # Mean difference
        sd = diff_bpm.std()                      # Standard deviation
        upper_loa = bias + 1.96 * sd             # Upper Limit of Agreement
        lower_loa = bias - 1.96 * sd             # Lower Limit of Agreement
        
        plt.figure(figsize=(10, 6))
        plt.scatter(mean_bpm, diff_bpm, alpha=0.4, color='#008080', edgecolors='none', s=20)
        
        # Draw statistical reference lines
        plt.axhline(bias, color='black', linestyle='-', linewidth=1.5)
        plt.axhline(upper_loa, color='red', linestyle='--', linewidth=1.2)
        plt.axhline(lower_loa, color='red', linestyle='--', linewidth=1.2)
        
        # Place text labels for the lines on the right side of the plot
        ax = plt.gca()
        x_text_pos = mean_bpm.max() + (mean_bpm.max() - mean_bpm.min()) * 0.02
        ax.text(x_text_pos, bias, f'Bias\n({bias:+.1f})', color='black', va='center', fontsize=9, fontweight='bold')
        ax.text(x_text_pos, upper_loa, f'+1.96 SD\n({upper_loa:+.1f})', color='red', va='center', fontsize=9)
        ax.text(x_text_pos, lower_loa, f'-1.96 SD\n({lower_loa:+.1f})', color='red', va='center', fontsize=9)
        
        plt.title(f'{main_title}: Bland-Altman Plot (HR)', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Mean of Measurements [ (Upper Arm + Wrist) / 2 ] (BPM)', fontsize=10)
        plt.ylabel('Difference [ Upper Arm - Wrist ] (BPM)', fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Stretch X-limit slightly so text on the right isn't cut off
        plt.xlim(mean_bpm.min() - 2, mean_bpm.max() + (mean_bpm.max() - mean_bpm.min()) * 0.15)
        plt.tight_layout()
        plt.savefig(os.path.join(run_folder, f"{pair_label}_HR_Bland_Altman.png"), dpi=300)
        plt.close()

    # ---------------------------------------------------------------------
    # PLOT 6: Cross-Correlation / Time-Lag Analysis (HR)
    # Quantifiziert den im Paper bisher nur visuell beschriebenen
    # "smoothing lag" der Handgelenk-Kurve gegenueber dem Oberarmsensor.
    # ---------------------------------------------------------------------
    best_lag, best_corr, corr_series = calculate_cross_correlation_lag(df_arm, df_wrist)

    if not corr_series.empty:
        plt.figure(figsize=(9, 5))
        plt.plot(corr_series.index, corr_series.values, color='#2ca02c', linewidth=1.8)
        plt.axvline(0, color='gray', linestyle=':', linewidth=1)
        if not np.isnan(best_lag):
            plt.axvline(best_lag, color='red', linestyle='--', linewidth=1.2,
                        label=f'Best lag = {best_lag:+d}s (r={best_corr:.2f})')
        plt.title(f'{main_title}: HR Cross-Correlation (Time Lag)', fontsize=12,
                  fontweight='bold', pad=15)
        plt.xlabel('Lag (s)  [positive = wrist lags behind upper arm]', fontsize=10)
        plt.ylabel('Correlation (r)', fontsize=10)
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(run_folder, f"{pair_label}_HR_Lag_Correlation.png"), dpi=300)
        plt.close()

    # ---------------------------------------------------------------------
    # PLOT 5: GPS Route Comparison -- mit KORREKT annotiertem Offset
    # (ersetzt die fehlerhafte Praxis, den Trial-3-Wert als generellen
    #  Wert zu zitieren -- hier steht immer der tatsaechliche Wert DIESES Laufs)
    # ---------------------------------------------------------------------
    mean_offset, max_offset, offset_series = calculate_gps_offset(df_arm, df_wrist)

    # -----------------------------------------------------------------
    # Distanz (OA/HG) + relativer Distanzfehler -- bisher nur als Rohwerte
    # in Table 1, ohne Fehlermetrik. Wird jetzt zusaetzlich zurueckgegeben,
    # damit main() daraus eine Cross-Run-Distanzfehler-Zusammenfassung bauen
    # kann (analog zu GPS-Offset und HR-Lag).
    # -----------------------------------------------------------------
    oa_distance_km = calculate_total_distance_km(df_arm)
    hg_distance_km = calculate_total_distance_km(df_wrist)
    distance_error_pct = calculate_distance_error(oa_distance_km, hg_distance_km)
    speed_r, speed_mae, speed_bias = calculate_speed_agreement(df_arm, df_wrist)

    plt.figure(figsize=(8, 8))
    plt.plot(df_arm['Longitude'], df_arm['Latitude'], color='#1f77b4', linewidth=2,
              label='Upper Arm (OA)', alpha=0.9)
    plt.plot(df_wrist['Longitude'], df_wrist['Latitude'], color='#d62728', linewidth=1.5,
              linestyle='--', label='Wrist (HG)', alpha=0.8)
    plt.scatter(df_arm['Longitude'].iloc[0], df_arm['Latitude'].iloc[0], color='green', s=80,
                zorder=5, label='Start')
    plt.scatter(df_arm['Longitude'].iloc[-1], df_arm['Latitude'].iloc[-1], color='black', s=80,
                marker='s', zorder=5, label='Ende')

    offset_txt = (f"Mean offset (this run): {mean_offset:.2f} m | Max: {max_offset:.1f} m"
                  if not np.isnan(mean_offset) else "Mean offset: n/a")
    plt.title(f'{main_title}: Route Comparison\n{offset_txt}', fontsize=11, fontweight='bold', pad=12)
    plt.xlabel('Longitude', fontsize=10)
    plt.ylabel('Latitude', fontsize=10)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, f"{pair_label}_Route_Comparison.png"), dpi=300)
    plt.close()

    return {
        "mean_offset": mean_offset, "max_offset": max_offset,
        "best_lag": best_lag, "best_lag_corr": best_corr,
        "hr_pearson_r": corr_coeff, "hr_mae": mae,
        "hr_bias": bias, "hr_upper_loa": upper_loa, "hr_lower_loa": lower_loa,
        "oa_distance_km": oa_distance_km, "hg_distance_km": hg_distance_km,
        "distance_error_pct": distance_error_pct,
        "speed_pearson_r": speed_r, "speed_mae_kmh": speed_mae, "speed_bias_kmh": speed_bias,
    }

# =========================================================================
# 4. MAIN LOGIC
# =========================================================================
if __name__ == "__main__":
    # Passend zu deiner Projektstruktur (relativ zu PyCode/, von wo aus du vk.py startest):
    #   data/csv/amazfit/Zepp_1..6.csv
    #   data/csv/flutter/Messung_1..6.csv
    zepp_files, csv_files = find_exact_pairs()

    run_labels = []
    mean_offsets = []
    max_offsets = []
    best_lags = []
    best_lag_corrs = []
    hr_pearson_rs = []
    hr_maes = []
    hr_biases = []
    hr_upper_loas = []
    hr_lower_loas = []
    oa_distances_km = []
    hg_distances_km = []
    distance_errors_pct = []
    speed_pearson_rs = []
    speed_maes_kmh = []
    speed_biases_kmh = []

    for i in range(6):
        df_arm = pd.read_csv(csv_files[i])
        df_wrist = pd.read_csv(zepp_files[i])
        
        # Process and reassign DataFrames
        dataframes = []
        for df in (df_arm, df_wrist):
            df['dt'] = pd.to_datetime(df['Timestamp']).dt.round('1s')
            if 'speed_native' not in df.columns:
                df = calculate_computed_speed(df)
                df['speed_native'] = df['computed_speed_kmh']
                df = df.drop(columns=['computed_speed_kmh'], errors='ignore')
            dataframes.append(df)
            
        # Unpack updated DataFrames
        df_arm, df_wrist = dataframes
        
        # Extract the date from the timestamp (Format: YYYY-MM-DD)
        date_string = df_arm['dt'].iloc[0].strftime('%Y-%m-%d') if not df_arm.empty else "Unknown Date"
        
        # Fully automatically generate all plots (inkl. Route + GPS-Offset fuer DIESEN Lauf)
        metrics = generate_all_plots(df_arm, df_wrist, f"Run_{i+1}", date_string)
        run_labels.append(f"Run {i+1}")
        mean_offsets.append(metrics["mean_offset"])
        max_offsets.append(metrics["max_offset"])
        best_lags.append(metrics["best_lag"])
        best_lag_corrs.append(metrics["best_lag_corr"])
        hr_pearson_rs.append(metrics["hr_pearson_r"])
        hr_maes.append(metrics["hr_mae"])
        hr_biases.append(metrics["hr_bias"])
        hr_upper_loas.append(metrics["hr_upper_loa"])
        hr_lower_loas.append(metrics["hr_lower_loa"])
        oa_distances_km.append(metrics["oa_distance_km"])
        hg_distances_km.append(metrics["hg_distance_km"])
        distance_errors_pct.append(metrics["distance_error_pct"])
        speed_pearson_rs.append(metrics["speed_pearson_r"])
        speed_maes_kmh.append(metrics["speed_mae_kmh"])
        speed_biases_kmh.append(metrics["speed_bias_kmh"])

        lag_txt = (f"{metrics['best_lag']:+.0f}s (r={metrics['best_lag_corr']:.2f})"
                   if not np.isnan(metrics["best_lag"]) else "n/a")
        print(f"Run {i+1} completely analyzed and all 6 plots generated. "
              f"(Mean GPS offset: {metrics['mean_offset']:.2f} m, Max: {metrics['max_offset']:.1f} m, "
              f"HR lag: {lag_txt}, HR r={metrics['hr_pearson_r']:.2f}, MAE={metrics['hr_mae']:.1f} bpm, "
              f"Distance error={metrics['distance_error_pct']:.1f}%, "
              f"Speed r={metrics['speed_pearson_r']:.2f}, Speed MAE={metrics['speed_mae_kmh']:.2f} km/h)")

    # =====================================================================
    # CROSS-RUN SUMMARY: der ehrliche Durchschnitt ueber ALLE Laeufe
    # (Fix fuer Gutachten-Punkt 4 -- nicht nur Trial 3 zitieren)
    # =====================================================================
    overall_mean = np.nanmean(mean_offsets)
    overall_max = np.nanmax(max_offsets)

    print("\n=== GPS OFFSET SUMMARY (all runs) ===")
    for label, m, mx in zip(run_labels, mean_offsets, max_offsets):
        print(f"  {label}: mean={m:.2f} m, max={mx:.1f} m")
    print(f"  --> Cross-run average (for Abstract): {overall_mean:.2f} m "
          f"(range {np.nanmin(mean_offsets):.2f}-{np.nanmax(mean_offsets):.2f} m, "
          f"overall max {overall_max:.1f} m)\n")

    with open("gps_offset_summary.txt", "w") as f:
        f.write("GPS Offset Summary (all N=6 runs)\n")
        f.write("==================================\n")
        for label, m, mx in zip(run_labels, mean_offsets, max_offsets):
            f.write(f"{label}: mean={m:.2f} m, max={mx:.1f} m\n")
        f.write(f"\nCross-run average mean offset: {overall_mean:.2f} m\n")
        f.write(f"Cross-run maximum offset: {overall_max:.1f} m\n")
        f.write("\nUse this cross-run average in the Abstract, NOT a single trial's value.\n")

    # Bar chart: offset per run + overall average line -- this is the plot
    # that should accompany/replace the misleading Figure 3a caption
    plt.figure(figsize=(9, 5.5))
    x = np.arange(len(run_labels))
    plt.bar(x, mean_offsets, color='#4c72b0', alpha=0.85, label='Mean GPS offset per run')
    plt.scatter(x, max_offsets, color='#d62728', zorder=5, label='Max GPS offset per run')
    plt.axhline(overall_mean, color='black', linestyle='--', linewidth=1.5,
                label=f'Cross-run average = {overall_mean:.2f} m')
    plt.xticks(x, run_labels)
    plt.ylabel('GPS Offset (m)')
    plt.title('GPS Offset: Upper Arm vs. Wrist -- all N=6 runs', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', framealpha=0.9)    
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("gps_offset_all_runs.png", dpi=300)
    plt.close()
    print("Saved corrected cross-run GPS offset summary: gps_offset_all_runs.png / gps_offset_summary.txt")

    # =====================================================================
    # CROSS-RUN SUMMARY: HR Cross-Correlation Lag ueber ALLE Laeufe
    # (quantifiziert den bisher nur visuell beschriebenen "smoothing lag")
    # =====================================================================
    valid_lags = [l for l in best_lags if not np.isnan(l)]
    if valid_lags:
        overall_mean_lag = float(np.mean(valid_lags))
        overall_median_lag = float(np.median(valid_lags))

        print("\n=== HR TIME-LAG SUMMARY (all runs) ===")
        for label, lag, corr in zip(run_labels, best_lags, best_lag_corrs):
            lag_txt = f"{lag:+.0f}s (r={corr:.2f})" if not np.isnan(lag) else "n/a"
            print(f"  {label}: lag={lag_txt}")
        print(f"  --> Cross-run mean lag: {overall_mean_lag:+.1f}s "
              f"(median {overall_median_lag:+.1f}s, "
              f"range {min(valid_lags):+.0f} to {max(valid_lags):+.0f}s)\n")

        with open("hr_lag_summary.txt", "w") as f:
            f.write("HR Cross-Correlation Lag Summary (all N=6 runs)\n")
            f.write("================================================\n")
            f.write("Positive lag = wrist HR lags behind upper-arm HR (seconds)\n\n")
            for label, lag, corr in zip(run_labels, best_lags, best_lag_corrs):
                lag_txt = f"{lag:+.0f}s (r={corr:.2f})" if not np.isnan(lag) else "n/a"
                f.write(f"{label}: lag={lag_txt}\n")
            f.write(f"\nCross-run mean lag: {overall_mean_lag:+.1f}s\n")
            f.write(f"Cross-run median lag: {overall_median_lag:+.1f}s\n")
            f.write(f"Range: {min(valid_lags):+.0f} to {max(valid_lags):+.0f}s\n")

        # Bar chart: lag per run + cross-run mean line
        plt.figure(figsize=(9, 5.5))
        x = np.arange(len(run_labels))
        plot_lags = [l if not np.isnan(l) else 0 for l in best_lags]
        plt.bar(x, plot_lags, color='#2ca02c', alpha=0.85, label='HR lag per run (s)')
        plt.axhline(overall_mean_lag, color='black', linestyle='--', linewidth=1.5,
                    label=f'Cross-run mean = {overall_mean_lag:+.1f}s')
        plt.axhline(0, color='gray', linestyle=':', linewidth=1)
        plt.xticks(x, run_labels)
        plt.ylabel('HR Lag (s) [positive = wrist behind upper arm]')
        plt.title('HR Time Lag: Upper Arm vs. Wrist -- all N=6 runs', fontsize=12,
                  fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig("hr_lag_all_runs.png", dpi=300)
        plt.close()
        print("Saved cross-run HR lag summary: hr_lag_all_runs.png / hr_lag_summary.txt")
    else:
        print("\nHR lag could not be computed for any run (insufficient overlapping data).")

    # =====================================================================
    # CROSS-RUN SUMMARY: HR Agreement (Pearson r, MAE, Bias, LoA) ueber
    # ALLE Laeufe -- bisher wurde im Paper nur Trial 3 (r=0.65) als
    # "illustratives Beispiel" gezeigt; hier der ehrliche Ueberblick ueber
    # alle 6 Laeufe, analog zum GPS-Offset- und HR-Lag-Summary.
    # =====================================================================
    valid_r = [v for v in hr_pearson_rs if not np.isnan(v)]
    valid_mae = [v for v in hr_maes if not np.isnan(v)]
    valid_bias = [v for v in hr_biases if not np.isnan(v)]
    if valid_r and valid_mae:
        mean_r, median_r = float(np.mean(valid_r)), float(np.median(valid_r))
        mean_mae = float(np.mean(valid_mae))
        mean_bias = float(np.mean(valid_bias)) if valid_bias else np.nan

        print("\n=== HR AGREEMENT SUMMARY (all runs) ===")
        for label, r, mae, bias, uloa, lloa in zip(
                run_labels, hr_pearson_rs, hr_maes, hr_biases, hr_upper_loas, hr_lower_loas):
            print(f"  {label}: r={r:.2f}, MAE={mae:.1f} bpm, Bias={bias:+.1f} bpm, "
                  f"LoA=[{lloa:+.1f}, {uloa:+.1f}]")
        print(f"  --> Cross-run mean r: {mean_r:.2f} (median {median_r:.2f}), "
              f"mean MAE: {mean_mae:.1f} bpm, mean Bias: {mean_bias:+.1f} bpm\n")

        with open("hr_agreement_summary.txt", "w") as f:
            f.write("HR Agreement Summary (all N=6 runs)\n")
            f.write("====================================\n")
            f.write("Bias = Upper Arm - Wrist (bpm); LoA = 95% Limits of Agreement\n\n")
            for label, r, mae, bias, uloa, lloa in zip(
                    run_labels, hr_pearson_rs, hr_maes, hr_biases, hr_upper_loas, hr_lower_loas):
                f.write(f"{label}: r={r:.2f}, MAE={mae:.1f} bpm, Bias={bias:+.1f} bpm, "
                        f"LoA=[{lloa:+.1f}, {uloa:+.1f}]\n")
            f.write(f"\nCross-run mean Pearson r: {mean_r:.2f}\n")
            f.write(f"Cross-run median Pearson r: {median_r:.2f}\n")
            f.write(f"Cross-run mean MAE: {mean_mae:.1f} bpm\n")
            f.write(f"Cross-run mean Bias: {mean_bias:+.1f} bpm\n")
            f.write("\nUse cross-run r/MAE (not Trial 3 alone) when characterizing overall agreement.\n")

        # Combined figure: r and MAE per run, two panels sharing the x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
        x = np.arange(len(run_labels))

        ax1.bar(x, [v if not np.isnan(v) else 0 for v in hr_pearson_rs],
                color='#6a0dad', alpha=0.85, label='Pearson r per run')
        ax1.axhline(mean_r, color='black', linestyle='--', linewidth=1.5,
                    label=f'Cross-run mean r = {mean_r:.2f}')
        ax1.set_ylabel('Pearson r')
        ax1.set_ylim(min(0, min(valid_r) - 0.1), 1.0)
        ax1.legend(loc='lower right', framealpha=0.9)
        ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
        ax1.set_title('HR Agreement: Upper Arm vs. Wrist -- all N=6 runs',
                       fontsize=12, fontweight='bold')

        ax2.bar(x, [v if not np.isnan(v) else 0 for v in hr_maes],
                color='#008080', alpha=0.85, label='MAE per run (bpm)')
        ax2.axhline(mean_mae, color='black', linestyle='--', linewidth=1.5,
                    label=f'Cross-run mean MAE = {mean_mae:.1f} bpm')
        ax2.set_ylabel('MAE (bpm)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(run_labels)
        ax2.legend(loc='upper right', framealpha=0.9)
        ax2.grid(True, axis='y', linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig("hr_agreement_all_runs.png", dpi=300)
        plt.close()
        print("Saved cross-run HR agreement summary: hr_agreement_all_runs.png / hr_agreement_summary.txt")
    else:
        print("\nHR agreement metrics could not be computed for any run.")

    # =====================================================================
    # CROSS-RUN SUMMARY: Distanz-Fehler (OA vs. HG) ueber ALLE Laeufe
    # Table 1 zeigte bisher nur die Rohwerte (OA/HG Distance in km) ohne
    # eine Fehlermetrik -- hier der relative Fehler pro Lauf + Mittelwert.
    # =====================================================================
    valid_dist_err = [v for v in distance_errors_pct if not np.isnan(v)]
    if valid_dist_err:
        mean_dist_err = float(np.mean(valid_dist_err))

        print("\n=== DISTANCE ERROR SUMMARY (all runs) ===")
        for label, oa, hg, err in zip(run_labels, oa_distances_km, hg_distances_km, distance_errors_pct):
            print(f"  {label}: OA={oa:.3f} km, HG={hg:.3f} km, Error={err:.1f}%")
        print(f"  --> Cross-run mean distance error: {mean_dist_err:.1f}%\n")

        with open("distance_error_summary.txt", "w") as f:
            f.write("Distance Error Summary (all N=6 runs)\n")
            f.write("======================================\n")
            f.write("Error = |OA - HG| / OA * 100 (Upper Arm treated as reference)\n\n")
            for label, oa, hg, err in zip(run_labels, oa_distances_km, hg_distances_km, distance_errors_pct):
                f.write(f"{label}: OA={oa:.3f} km, HG={hg:.3f} km, Error={err:.1f}%\n")
            f.write(f"\nCross-run mean distance error: {mean_dist_err:.1f}%\n")

        plt.figure(figsize=(9, 5.5))
        x = np.arange(len(run_labels))
        plt.bar(x, distance_errors_pct, color='#e07b39', alpha=0.85, label='Distance error per run (%)')
        plt.axhline(mean_dist_err, color='black', linestyle='--', linewidth=1.5,
                    label=f'Cross-run mean = {mean_dist_err:.1f}%')
        plt.xticks(x, run_labels)
        plt.ylabel('Distance Error (%)  [relative to Upper Arm]')
        plt.title('GPS Distance Error: Upper Arm vs. Wrist -- all N=6 runs',
                  fontsize=12, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig("distance_error_all_runs.png", dpi=300)
        plt.close()
        print("Saved cross-run distance error summary: distance_error_all_runs.png / distance_error_summary.txt")
    else:
        print("\nDistance error could not be computed for any run.")

    # =====================================================================
    # CROSS-RUN SUMMARY: Speed Agreement (Pearson r, MAE, Bias) ueber ALLE
    # Laeufe -- bisher wurde Speed nur geplottet (Speed Profile), nie
    # quantifiziert. Analog zur HR-Agreement-Summary oben.
    # =====================================================================
    valid_speed_r = [v for v in speed_pearson_rs if not np.isnan(v)]
    valid_speed_mae = [v for v in speed_maes_kmh if not np.isnan(v)]
    if valid_speed_r and valid_speed_mae:
        mean_speed_r = float(np.mean(valid_speed_r))
        median_speed_r = float(np.median(valid_speed_r))
        mean_speed_mae = float(np.mean(valid_speed_mae))
        valid_speed_bias = [v for v in speed_biases_kmh if not np.isnan(v)]
        mean_speed_bias = float(np.mean(valid_speed_bias)) if valid_speed_bias else np.nan

        print("\n=== SPEED AGREEMENT SUMMARY (all runs) ===")
        for label, r, mae, bias in zip(run_labels, speed_pearson_rs, speed_maes_kmh, speed_biases_kmh):
            print(f"  {label}: r={r:.2f}, MAE={mae:.2f} km/h, Bias={bias:+.2f} km/h")
        print(f"  --> Cross-run mean r: {mean_speed_r:.2f} (median {median_speed_r:.2f}), "
              f"mean MAE: {mean_speed_mae:.2f} km/h, mean Bias: {mean_speed_bias:+.2f} km/h\n")

        with open("speed_agreement_summary.txt", "w") as f:
            f.write("Speed Agreement Summary (all N=6 runs)\n")
            f.write("=======================================\n")
            f.write("Bias = Upper Arm (smoothed) - Wrist (km/h)\n\n")
            for label, r, mae, bias in zip(run_labels, speed_pearson_rs, speed_maes_kmh, speed_biases_kmh):
                f.write(f"{label}: r={r:.2f}, MAE={mae:.2f} km/h, Bias={bias:+.2f} km/h\n")
            f.write(f"\nCross-run mean Pearson r: {mean_speed_r:.2f}\n")
            f.write(f"Cross-run median Pearson r: {median_speed_r:.2f}\n")
            f.write(f"Cross-run mean MAE: {mean_speed_mae:.2f} km/h\n")
            f.write(f"Cross-run mean Bias: {mean_speed_bias:+.2f} km/h\n")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
        x = np.arange(len(run_labels))

        ax1.bar(x, [v if not np.isnan(v) else 0 for v in speed_pearson_rs],
                color='#c44e52', alpha=0.85, label='Pearson r per run')
        ax1.axhline(mean_speed_r, color='black', linestyle='--', linewidth=1.5,
                    label=f'Cross-run mean r = {mean_speed_r:.2f}')
        ax1.set_ylabel('Pearson r')
        ax1.set_ylim(min(0, min(valid_speed_r) - 0.1), 1.0)
        ax1.legend(loc='lower right', framealpha=0.9)
        ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
        ax1.set_title('Speed Agreement: Upper Arm vs. Wrist -- all N=6 runs',
                       fontsize=12, fontweight='bold')

        ax2.bar(x, [v if not np.isnan(v) else 0 for v in speed_maes_kmh],
                color='#dd8452', alpha=0.85, label='MAE per run (km/h)')
        ax2.axhline(mean_speed_mae, color='black', linestyle='--', linewidth=1.5,
                    label=f'Cross-run mean MAE = {mean_speed_mae:.2f} km/h')
        ax2.set_ylabel('MAE (km/h)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(run_labels)
        ax2.legend(loc='upper right', framealpha=0.9)
        ax2.grid(True, axis='y', linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig("speed_agreement_all_runs.png", dpi=300)
        plt.close()
        print("Saved cross-run speed agreement summary: speed_agreement_all_runs.png / speed_agreement_summary.txt")
    else:
        print("\nSpeed agreement metrics could not be computed for any run.")