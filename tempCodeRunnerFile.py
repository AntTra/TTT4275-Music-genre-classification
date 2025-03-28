#Plot PDF of genres and features 
# plt.figure(figsize=(14, 12))
# for i, feature in enumerate(selected_features):
#     ax = plt.subplot(2, 2, i+1)
#     for genre in selected_genres:
#         subset = data_filtered[data_filtered["Genre"] == genre]
#         sns.kdeplot(subset[feature], label=genre, fill=True, common_norm=False, ax=ax)
#         ax.set_title("")
#     #plt.title(f"PDF of {feature}")
#     #plt.xlabel(feature)
#     plt.ylabel("Density")
# #plt.tight_layout()
# plt.legend()
# plt.show()