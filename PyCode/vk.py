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
def find_exact_pairs():
    gpx_ordered = [os.path.join("data", f"Zepp_{i}.gpx") for i in range(1, 7)]
    csv_ordered = [os.path.join("data", "csv", "flutter", f"Messung_{i}.csv") for i in range(1, 7)]
    return gpx_ordered, csv_ordered

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

def calculate_computed_speed(df):
    df = df.sort_values('dt').reset_index(drop=True)
    speeds = [0.0]
    for i in range(1, len(df)):
        dt = (df.loc[i, 'dt'] - df.loc[i-1, 'dt']).total_seconds()
        if dt > 0: speeds.append((haversine(df.loc[i-1, 'Latitude'], df.loc[i-1, 'Longitude'], df.loc[i, 'Latitude'], df.loc[i, 'Longitude']) / dt) * 3.6)
        else: speeds.append(0.0)
    df['computed_speed_kmh'] = speeds
    return df

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

# =========================================================================
# 4. MAIN LOGIC
# =========================================================================
if __name__ == "__main__":
    gpx_files, csv_files = find_exact_pairs()
    output_dir = os.path.join("data", "csv", "amazfit")
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(6):
        zepp_csv = os.path.join(output_dir, f"Zepp_{i+1}.csv")
        convert_gpx_to_csv(gpx_files[i], zepp_csv)
        
        df_arm = pd.read_csv(csv_files[i])
        df_wrist = pd.read_csv(zepp_csv)
        
        # Process and reassign DataFrames
        dataframes = []
        for df, path in [(df_arm, csv_files[i]), (df_wrist, zepp_csv)]:
            df['dt'] = pd.to_datetime(df['Timestamp']).dt.round('1s')
            if 'speed_native' not in df.columns:
                df = calculate_computed_speed(df)
                df['speed_native'] = df['computed_speed_kmh']
                df = df.drop(columns=['computed_speed_kmh'], errors='ignore')
            df.to_csv(path, index=False)
            dataframes.append(df)
            
        # Unpack updated DataFrames
        df_arm, df_wrist = dataframes
        
        # Extract the date from the timestamp (Format: YYYY-MM-DD)
        date_string = df_arm['dt'].iloc[0].strftime('%Y-%m-%d') if not df_arm.empty else "Unknown Date"
        
        # Fully automatically generate all four plots
        generate_all_plots(df_arm, df_wrist, f"Run_{i+1}", date_string)        
        print(f"Run {i+1} completely analyzed and all 4 plots generated.")