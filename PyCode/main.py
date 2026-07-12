import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ==========================================
# 1. ECHTE DATEN EINLESEN (Automatisch)
# ==========================================
# Anstatt Daten zu generieren, lesen wir deine CSV-Dateien ein.
# 'parse_dates=True' ist der Zauberbefehl: Er sagt Pandas, dass es die 
# Spalte mit der Uhrzeit automatisch als echtes Zeit-Format erkennen soll.

# Ersetze "Zeitstempel" durch den tatsächlichen Spaltennamen in deiner CSV
df_s1 = pd.read_csv("lauf_1_handgelenk.csv", index_col="Zeitstempel", parse_dates=True)
df_s2 = pd.read_csv("lauf_1_oberarm.csv", index_col="Zeitstempel", parse_dates=True)

# Ab hier muss NICHTS MEHR GEÄNDERT WERDEN! 
# Python sucht sich die Start- und Endzeiten jetzt vollautomatisch aus deinen Dateien.

# ==========================================
# 2. LOGIK: START UND ENDE ANGLEICHEN (Vollautomatisch)
# ==========================================
neuer_start = max(df_s1.index.min(), df_s2.index.min())
neues_ende = min(df_s1.index.max(), df_s2.index.max())

df_s1_clean = df_s1.loc[neuer_start : neues_ende].copy()
df_s2_clean = df_s2.loc[neuer_start : neues_ende].copy()

print(f"Lauf startete automatisch am: {neuer_start}")
print(f"Lauf endete automatisch am: {neues_ende}")