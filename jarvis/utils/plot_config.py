from matplotlib import pyplot as plt

# 1) Light grey grid lines everywhere
plt.rcParams['axes.grid']       = True
plt.rcParams['grid.color']      = 'lightgrey'
plt.rcParams['grid.linestyle']  = '-'
plt.rcParams['grid.linewidth']  = 0.5
plt.rcParams['grid.alpha']      = 0.8

# 2) Default font sizes
plt.rcParams['font.size']       = 12    # general text
plt.rcParams['axes.titlesize']  = 14    # subplot title
plt.rcParams['axes.labelsize']  = 12    # x/y labels
plt.rcParams['xtick.labelsize'] = 10    # x-tick labels
plt.rcParams['ytick.labelsize'] = 10    # y-tick labels
plt.rcParams['legend.fontsize'] = 11    # legend text
