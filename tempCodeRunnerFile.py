selected_genres = ["pop", "disco", "metal", "classical"]#, "hiphop", "reggae", "blues", "rock", "jazz", "country"]
# # Filter data by selected genres
# data_filtered = df[df["Genre"].isin(selected_genres)]

# # Calculate summary statistics grouped by Genre
# summary_stats = data_filtered.groupby("Genre")[selected_features].describe()
# print("Summary Statistics by Genre:")
# print(summary_stats)
# print(knn.covariance())