import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import math
# ---------------------------------
# Data Loading and Preprocessing
# ---------------------------------
def load_data(filepath):
    df = pd.read_csv(filepath, delimiter='\t')
    df.columns = df.columns.str.strip()  # Remove extra whitespace from column names
    return df

filepath = 'Classification music/GenreClassData_30s.txt'
df = load_data(filepath)

# Selected features and genres 
#selected_features = ["spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", "spectral_rolloff_mean", "spectral_rolloff_var", "spectral_contrast_mean", "spectral_contrast_var", "spectral_flatness_mean", "spectral_flatness_var"]
#selected_features = ["chroma_stft_1_mean", "chroma_stft_2_mean", "chroma_stft_3_mean", "chroma_stft_4_mean", "chroma_stft_5_mean", "chroma_stft_6_mean", "chroma_stft_7_mean", "chroma_stft_8_mean", "chroma_stft_9_mean", "chroma_stft_10_mean", "chroma_stft_11_mean", "chroma_stft_12_mean"]
#selected_features = ["mfcc_1_mean", "mfcc_2_mean", "mfcc_3_mean", "mfcc_4_mean", "mfcc_5_mean", "mfcc_6_mean", "mfcc_7_mean", "mfcc_8_mean", "mfcc_9_mean", "mfcc_10_mean", "mfcc_11_mean", "mfcc_12_mean"]
selected_features = ["mfcc_1_std", "mfcc_2_std", "mfcc_3_std", "mfcc_4_std", "mfcc_5_std", "mfcc_6_std", "mfcc_7_std", "mfcc_8_std", "mfcc_9_std", "mfcc_10_std", "mfcc_11_std", "mfcc_12_std"]
#selected_features = ["tempo"] 
#selected_features = ["mfcc_1_mean", "spectral_rolloff_mean", "spectral_centroid_mean", "tempo"]
#selected_genres = ["reggae", "jazz", "disco", "classical", "metal", "country"]

# Genres:   "pop", "disco", "metal", "classical", "hiphop", "reggae", "blues", "rock", "jazz", "country"]
selected_genres = ["pop", "metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"] # All genres
#selected_genres = ["reggae", "jazz", "disco", "classical", "metal", "country"]
#selected_genres = ["reggae"]

data_filtered = df[df["Genre"].isin(selected_genres)]

# ---------------------------------
# Plotting Functions
# ---------------------------------
def plot_pdf():
    """
    Plot the 1D KDE (PDF) for each selected feature,
    with curves for each genre overlaid.
    """
    n_features = len(selected_features)
    # Adjust the layout based on how many features you want to plot
    n_columns = 4
    n_rows = math.ceil(n_features / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4, n_rows * 3))
    
    # Ensure axes is a flat array of Axes objects
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    
    for i, feature in enumerate(selected_features):
        ax = axes[i]
        for genre in selected_genres:
            subset = data_filtered[data_filtered["Genre"] == genre]
            sns.kdeplot(subset[feature], label=genre, fill=True, common_norm=False, ax=ax)
        ax.set_title(f"PDF of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_contour():
    """
    Plot a 2D KDE contour plot for a specified pair of features,
    with one custom legend entry per genre.
    """
    # Define the features to plot
    feature_x = "spectral_centroid_mean"
    feature_y = "mfcc_1_mean"
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a color palette with one color per genre
    palette = sns.color_palette("viridis_r", len(selected_genres))
    
    # Plot the contour for each genre using a unique color, without setting a label.
    for i, genre in enumerate(selected_genres):
        subset = data_filtered[data_filtered["Genre"] == genre]
        sns.kdeplot(
            x=subset[feature_x],
            y=subset[feature_y],
            #cmap='viridis_r',
            levels=5,             
            fill=True,           
            common_norm=False,
            alpha=0.5,
            color=palette[i],
            label=None
        )
        plt.scatter(
            subset[feature_x], 
            subset[feature_y], 
            color=palette[i], 
            alpha=0.3, 
            s=10  # Adjust size for better visibility
        )
    
    # Remove any automatically generated legend entries, if present
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    # Create custom legend handles for each genre
    handles = [mpatches.Patch(color=palette[i], label=genre) 
               for i, genre in enumerate(selected_genres)]
    
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f"Contour Plot of {feature_x} vs. {feature_y} by Genre")
    ax.legend(handles=handles, title="Genre")
    plt.show()

def plot_cov_per_genre(data, genres, features):
    """
    For each genre in `genres`, compute and plot the covariance
    matrix of `features` restricted to that genre.
    """
    import math

    n_genres = len(genres)
    n_cols = 4
    n_rows = math.ceil(n_genres / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 3),
        constrained_layout=True
    )
    # flatten axes array
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax, genre in zip(axes_flat, genres):
        subset = data[data["Genre"] == genre]
        cov = subset[features].cov()
        sns.heatmap(
            cov,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=False,
            ax=ax
        )
        ax.set_title(f"{genre} covariance")
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    # turn off any extra axes
    for ax in axes_flat[n_genres:]:
        ax.axis("off")

    plt.suptitle("Per-Genre Covariance Heatmaps", y=1.02)
    plt.show()

# ---------------------------------
# Justification of Mahalanobian usage
# ---------------------------------
def plot_corr():
    cov = df[selected_features].cov()
    print("Covariance matrix:")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cov, annot=True, cmap='coolwarm')
    plt.title("Covariance Matrix Heatmap")
    plt.show()

# ----------------------------------
# Main: Select which plot(s) to display
# ---------------------------------
if __name__ == "__main__":
    show_pdf_plot = True
    show_contour_plot = False
    show_cov_plot = True
    show_corr_plot = False

    if show_pdf_plot:
        plot_pdf()
    if show_contour_plot:
        plot_contour()
    if show_cov_plot:
        plot_cov_per_genre(data_filtered, selected_genres, selected_features)
    if show_corr_plot:
        plot_corr()