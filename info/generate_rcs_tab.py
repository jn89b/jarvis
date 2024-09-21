"""
Parse out the information 

Columns are relative azmith positions:
from 0 to 360 in 2 degree increments

Rows are relative elevation positions:
from -80 to 80 in 10 degree increments
"""
import pandas as pd

# plot as a spider plot
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


info_dir = 'info/'
filename = 'plane_90s_plane_sig.csv'
df = pd.read_csv(info_dir+filename, header=None)

#get the entire 8th row
rcs_vals = df.iloc[8]
#interpolate the values to project from 0 to 360
azimuths = np.linspace(0, 360, len(rcs_vals))
interp_func = interp1d(azimuths, rcs_vals)

azimuths = np.arange(0, 360, 1)
rcs_vals = interp_func(azimuths)



#generate a hashtable of the values
azmith_keys = [str(az) for az in azimuths]
rcs_hash = dict(zip(azmith_keys, rcs_vals))
#pickled the hash
import pickle
with open('rcs_hash.pkl', 'wb') as f:
    pickle.dump(rcs_hash, f)

#get the 0
print(rcs_hash['0'])

# Create a spider plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Set the azimuthal angle
azimuths = np.linspace(0, 2 * np.pi, 360, endpoint=False)

# Plot the radar cross section values
ax.plot(azimuths, rcs_vals)

plt.show()