import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# ==========================================
# 1. DEFINE THE STUDY PARAMETERS
# ==========================================
delta = 5.0      # Expected minimal difference in bpm (Minimal Detectable Difference)
sigma = 6.0      # Standard deviation of the differences in bpm
alpha = 0.05     # Significance level (5%)
power_ziel = 80  # Target statistical power in %

# Z-Wert für das zweiseitige Alpha (1.96 für alpha=0.05)
z_alpha = norm.ppf(1 - alpha/2) 

# ==========================================
# 2. CALCULATING THE CURVE
# ==========================================
# We calculate the power for sample sizes (N) ranging from 4 to 30 trials
n_values = np.arange(4, 31, 1)
power_values = []

# Iterative calculation of statistical power for each N
for n in n_values:
    # Umstellung der Power-Formel nach Z_beta: Z_beta = sqrt(N) * (Delta / Sigma) - Z_alpha
    z_beta = np.sqrt(n) * (delta / sigma) - z_alpha
    # Umwandlung des Z-Wertes in Wahrscheinlichkeit (Power in %)
    power = norm.cdf(z_beta) * 100
    power_values.append(power)

# Calculating the exact power for your current study (N=6)
current_n = 6
current_power = norm.cdf(np.sqrt(current_n) * (delta / sigma) - z_alpha) * 100

# ==========================================
# 3. CREATE AND FORMAT DIAGRAM
# ==========================================
plt.figure(figsize=(10, 6))

# Plot the main curve
plt.plot(n_values, power_values, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, label='Calculated Statistical Power')

# Draw threshold lines (80% power and required N=13)
plt.axhline(y=power_ziel, color='#d62728', linestyle='--', linewidth=1.5, label=f'Target: {power_ziel}% Statistical Power')
plt.axvline(x=13, color='#2ca02c', linestyle=':', linewidth=2, label='Required Sample Size $n \\approx 13$')

# Mark the current data point (N=6)
plt.plot(current_n, current_power, marker='s', markersize=10, color='darkorange',
         label=f'Current Pilot Study ($N={current_n}$, Power $\\approx$ {current_power:.1f}%)')

# Text annotation at point N=6
plt.annotate(f'{current_power:.1f}%',
             xy=(current_n, current_power),
             xytext=(current_n + 0.8, current_power - 8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontsize=11, color='darkorange', fontweight='bold')

# Axis labels and title
plt.title('Post-hoc Power Analysis for Wearable Validation', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Sample size (number of runs $N$)', fontsize=12)
plt.ylabel('Statistical power (%)', fontsize=12)

# Adjust axis scaling
plt.xticks(np.arange(4, 32, 2))
plt.yticks(np.arange(0, 101, 10))

# Grid and legend
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True, borderpad=1)

# Optimize layout and save as PNG for LaTeX
plt.tight_layout()
plt.savefig('power_analysis_curve.png', dpi=300, bbox_inches='tight')

# Display the diagram (optional)
print("Diagram saved as 'power_analysis_curve.png' in the current directory.")
plt.show()  # Remove the hash if you want a window to open when running