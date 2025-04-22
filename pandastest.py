import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('Classification music/GenreClassData_30s.txt', delimiter='\t')
#print(df.columns)
#print(df['Track ID'])

[plt.scatter(item[0], item[1], s=100, color=i) for i in dataset for item in dataset[i]]
plt.scatter(new_features[0], new_features[1], s=200, color=result[0]);
