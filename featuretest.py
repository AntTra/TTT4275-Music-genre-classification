import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Classification music/GenreClassData_5s.txt', delimiter='\t')

# Whitespace removal
df.columns = df.columns.str.strip()

# MFCC for timbral characteristics
# Spectral features for "brightness" and "sharpness"
# Tempo for "speed" and "rhythm"
# RMSE for "loudness" and "energy"
# Chroma features for "harmony" and "tonality"

# Selected features for analysis
selected_features = ["spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", "spectral_rolloff_mean", "spectral_rolloff_var", "spectral_contrast_mean", "spectral_contrast_var", "spectral_flatness_mean", "spectral_flatness_var"]
#selected_features = ["chroma_stft_1_mean", "chroma_stft_2_mean", "chroma_stft_3_mean", "chroma_stft_4_mean", "chroma_stft_5_mean", "chroma_stft_6_mean", "chroma_stft_7_mean", "chroma_stft_8_mean", "chroma_stft_9_mean", "chroma_stft_10_mean", "chroma_stft_11_mean", "chroma_stft_12_mean"]
#selected_features = ["mfcc_1_mean", "mfcc_2_mean", "mfcc_3_mean", "mfcc_4_mean", "mfcc_5_mean", "mfcc_6_mean", "mfcc_7_mean", "mfcc_8_mean", "mfcc_9_mean", "mfcc_10_mean", "mfcc_11_mean", "mfcc_12_mean"]
#selected_features = ["mfcc_1_std", "mfcc_2_std", "mfcc_3_std", "mfcc_4_std", "mfcc_5_std", "mfcc_6_std", "mfcc_7_std", "mfcc_8_std", "mfcc_9_std", "mfcc_10_std", "mfcc_11_std", "mfcc_12_std"]
#selected_features = ["tempo"] 
sel_feature = ["mfcc_1_mean"]
# Genres:   "pop", "disco", "metal", "classical", "hiphop", "reggae", "blues", "rock", "jazz", "country"]
selected_genres = ["pop", "disco", "metal", "hiphop", "reggae",  "rock", "jazz", "country"]
#selected_genres = ["pop", "disco", "metal", "classical"]



n_features = len(selected_features)
n_columns = 4
n_rows = (n_features + n_columns - 1) // n_columns
fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4 , n_rows * 3))
axes = axes.flatten()

# Filter data by selected genres
data_filtered = df[df["Genre"].isin(selected_genres)]


for i, feature in enumerate(selected_features):
    ax = axes[i]
    for genre in selected_genres:
        subset = data_filtered[data_filtered["Genre"] == genre]
        sns.kdeplot(subset[feature], label=genre, fill=True, common_norm=False, ax=ax)
    ax.set_title(f"PDF of {feature}")
    #ax.set_title("")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
plt.tight_layout()
ax.legend()
for j in range(len(selected_features), len(axes)):
    fig.delaxes(axes[j])
plt.show()


# Define the features you want to plot
feature_x = "mfcc_2_std"
feature_y = "mfcc_1_std"

data_filtered = df[df["Genre"].isin(selected_genres)]

# plt.figure(figsize=(10, 8))
# # Loop over each genre and plot the scatter plot
# for i, feature in enumerate(selected_features):
#     ax = axes[i]
#     a = 0
#     for genre in selected_genres:
        
#         subset = data_filtered[data_filtered["Genre"] == genre]
#         #sns.clusterplot(x=feature_x, y=feature_y, data=subset, label=genre, alpha=0.5)
#         #sns.clustermap(subset.pivot(index=feature_x, columns=feature_y, values=subset.values),cmap="viridis",standard_scale=1)#, data=subset, label=genre, alpha=0.5)
#         sns.histplot(x=feature_x, y=feature_y, data=subset, label=genre, alpha=0.5)
#         #plt.scatter(subset[feature_x], subset[feature_y], label=genre, alpha=0.5)
    
#     plt.xlabel(feature_x)
#     plt.ylabel(feature_y)
# plt.title(f"Scatter Plot of {feature_x} vs. {feature_y}")
# plt.legend()
# plt.show()

n_genres = len(selected_genres)
n_columns = 4
n_rows = (n_genres + n_columns - 1) // n_columns

fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4, n_rows * 3))
axes = axes.flatten()

for i, genre in enumerate(selected_genres):
    subset = data_filtered[data_filtered["Genre"] == genre]
    axes[i].hist2d(subset[feature_x], subset[feature_y], bins=30, cmap='viridis')
    axes[i].set_title(genre)
    axes[i].set_xlabel(feature_x)
    axes[i].set_ylabel(feature_y)

# Remove any extra subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
    
plt.tight_layout()
plt.show()