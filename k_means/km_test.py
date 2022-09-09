import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import k_means as km # <-- Your implementation

sns.set_style('darkgrid')

# ## Data set 1

# data_1 = pd.read_csv('data_1.csv')
# data_1.describe().T

# # Visualize
# plt.figure(figsize=(5, 5))
# sns.scatterplot(x='x0', y='x1', data=data_1)
# plt.xlim(0, 1); plt.ylim(0, 1)

# # Fit Model 
# X = data_1[['x0', 'x1']].to_numpy() 
# model_1 = km.KMeans(2) # <-- Should work with default constructor  
# model_1.fit(X)

# # Compute Silhouette Score 
# z = model_1.predict(X)
# print(f'Silhouette Score: {km.euclidean_silhouette(X, z) :.3f}')
# print(f'Distortion: {km.euclidean_distortion(X, z) :.3f}')

# # Plot cluster assignments
# C = model_1.get_centroids()
# K = len(C)
# _, ax = plt.subplots(figsize=(5, 5), dpi=100)
# sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K), palette='tab10', data=data_1[['x0', 'x1']], ax=ax);
# sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
# ax.legend().remove()

## Data set 2

data_2 = pd.read_csv('data_2.csv')
data_2.describe().T

plt.figure(figsize=(5, 5))
sns.scatterplot(x='x0', y='x1', data=data_2);

# Fit Model 
X = data_2[['x0', 'x1']].to_numpy()
model_2 = km.KMeans(10,2,10)  # <-- Feel free to add hyperparameters 
model_2.fit(X)

# Compute Silhouette Score 
z = model_2.predict(X)
print(f'Distortion: {km.euclidean_distortion(X, z) :.3f}')
print(f'Silhouette Score: {km.euclidean_silhouette(X, z) :.3f}')

# Plot cluster assignments
C = model_2.get_centroids()
K = len(C)
_, ax = plt.subplots(figsize=(5, 5), dpi=100)
sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K), palette='tab10', data=data_2[['x0', 'x1']], ax=ax);
sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
ax.legend().remove()