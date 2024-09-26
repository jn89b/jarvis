import numpy as np
import matplotlib.pyplot as plt

"""
Typical SNR Threshold:
    -10 to 13dB

    - High Confidence detection: 15 - 20 dB:Military

    - Lower-Power systems: 6-8db

"""

# Constants based on the previous explanation and given RCS
Pt = 500  # Transmitted power in watts
G_db = 10   # Antenna gain in linear scale
G = 10**(G_db/10)  # Antenna gain in dB
lambda_radar = 0.03  # Wavelength in meters
k = 1.38e-23  # Boltzmann's constant in J/K
Ts = 900  # System noise temperature in Kelvin
Bn = 1e6  # Noise bandwidth in Hz
L_db = 8  # Losses in linear scale
L = 10**(L_db/10)  # Losses in dB
RCS = 0.0731  # Given RCS value (linear_db)
SNR_THRESHOLD_DB = 10  # SNR threshold in dB
SNR_THRESHOLD_DB_LINEAR = 10**(SNR_THRESHOLD_DB/10)  # SNR threshold in linear scale
print(SNR_THRESHOLD_DB_LINEAR)

# Distance values (meters)
distances = np.linspace(0, 500, 100)  # From 100 meters to 10,000 meters

# Radar equation for S/N
SN = (Pt * G**2 * lambda_radar**2 * RCS) / ((4 * np.pi)**3 * distances**4 * k * Ts * Bn * L)

# Radar detection probability based on the given detection model equation
#radar_prob_detection = 1 / (1 + (c2 * distances**4 / SN)**c1)
radar_prob_detection = 1 - np.exp(-SN/SNR_THRESHOLD_DB_LINEAR)

# Plotting distance vs probability of detection
plt.figure(figsize=(10, 6))
plt.plot(distances, radar_prob_detection, label="Probability of Detection")
plt.xlabel('Distance (meters)')
plt.ylabel('Probability of Detection')
plt.title('Distance vs Probability of Detection for Radar System')
plt.grid(True)
plt.legend()
plt.show()
