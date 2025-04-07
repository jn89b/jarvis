import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns

# General figure settings
# plt.rcParams['figure.figsize'] = (3.25, 2.5)  # Adjust to your column width
# plt.rcParams['figure.dpi'] = 300             # High DPI for print quality

# Font sizes
plt.rcParams['font.size'] = 12                # Base font size
plt.rcParams['axes.titlesize'] = 18          # Axis title size
plt.rcParams['axes.labelsize'] = 16           # Axis label size
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.labelsize'] = 18

# set the color pallete
sns.set_palette("husl")
plt.close('all')

# 1) Define the labels for the models and columns
models = [
    "CKAE",
    "LSTM-Encoder-Decoder",
    "ITB-LSTM",
    "CNN-LSTM",
    "GA-FHT-Etman",
    "AttConv-LSTM",    
    "Predictformer"
]

column_labels = ["0.5", "1.0", "1.5", "2.0", "2.5"]
predictformer_metrics = pkl.load(open("small_model_metrics.pkl", "rb"))
lstm_metrics = pkl.load(open("lstm_metrics.pkl", "rb"))
# set a color for each of the models
colors = sns.color_palette("husl", len(models))
color_key = dict(zip(models, colors))
# each get the first 5 elements and average
lstm_bins = []
overall_bins = []

# %%
## This is the LSTM-Encoder-Decoder
mse_bins = np.array(lstm_metrics["slice_mse"])
mse_bins = mse_bins[:, :5]/2
lstm_bins.append(np.mean(mse_bins, axis=0))

for i, agent in enumerate(predictformer_metrics):
    mse_bins = np.array(agent["slice_mse"])
    mse_bins = mse_bins[:, :5]
    overall_bins.append(np.mean(mse_bins, axis=0))

predictformer_sample = overall_bins[0]
# 2) Define the data as a list of lists (rows correspond to models)
data = [
    [6.95190, 11.12336, 30.0703, 46.6668, 65.5497],   # CKAE [48]
    [5.42246, 7.80415, 14.4761, 17.83114, 23.23122],  # ITB-LSTM [56]
    [2.99728, 5.65739, 10.29234, 11.62909, 14.80631], # CNN-LSTM [29]
    [1.60886, 1.95828, 8.61368, 10.10829, 15.68474],  # GA-FHT-Etman [45]
    [1.04976, 2.46361, 6.90185, 7.51149, 13.93615],   # AttConv-LSTM
]

#data.extend(list(lstm_bins))
# Insert the LSTM-Encoder-Decoder data as the second row (index 1)
data.insert(1, list(lstm_bins[0]))
data.extend([list(predictformer_sample)])

# 3) Create a grouped bar chart
x = np.arange(len(column_labels))  # positions for each column group
width = 0.12  # width of each bar

fig, ax = plt.subplots(figsize=(16, 10))
# add horizontal grid lienes
# Plot each model’s data, shifting each bar group horizontally
for i, model in enumerate(models):
    print(i)
    # Shift by i * width so bars don’t overlap
    ax.bar(x + i*width, data[i], width, label=model, color=color_key[model])
    # annotate the bars
    for j, value in enumerate(data[i]):
        ax.text(j + i*width, value + 1, f"{value:.1f}", ha='center', va='bottom',
                fontsize=14)
        
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)


# 4) Labeling and styling
ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_title('Mean Absolute Error Comparison (Lower is Better)')
ax.set_xticks(x + width*(len(models)-1)/2)
ax.set_xlabel('Projection Time (s)')
ax.set_xticklabels(column_labels)
ax.legend()

fig.tight_layout()
plt.savefig("prediction_model_comparison.svg")

# Using a scatter plot to compare the models 
fig, ax = plt.subplots()
line_types = ['-', '--', '-.', ':', 'solid', 'dotted']
shapes = ['o', 's', 'v', '^', 'x', 'd']
for i, model in enumerate(models):
    ax.plot(column_labels, data[i], label=model, color=color_key[model], linestyle=line_types[i])
    ax.scatter(column_labels, data[i], color=color_key[model], marker=shapes[i], 
               label=model)
ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_title('Comparison of Models')
ax.set_xlabel('Projection Time (s)')
ax.legend()
fig.tight_layout()
plt.savefig("prediction_model_comparison_line.svg")

## -------------------- BENCHMARK AGAINST ATTENTION LSTM --------------------
# Let's just benchmark the predictformer against the lattconv-lstm
final_comparison = data[-2:]
models = [
    "AttConv-LSTM",
    "Predictformer"
]

# get the color for the models
fig, ax = plt.subplots()
x = np.arange(len(column_labels))
width = 0.35
for i, model in enumerate(models):
    ax.bar(x + i*width, final_comparison[i], width, 
           label=model, color=color_key[model])

ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_title('Comparison of Models')
ax.set_xticks(x + width*(len(models)-1)/2)
ax.set_xlabel('Projection Time (s)')
ax.set_xticklabels(column_labels)
ax.legend()

# save the figure
plt.savefig("predictformer_vs_attconv_lstm.svg")
plt.tight_layout()
plt.show()
