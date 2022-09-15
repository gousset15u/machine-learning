from statistics import mode
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
model_2 = km.KMeans(10,2)  # <-- Feel free to add hyperparameters 
model_2.fit(X)

# Compute Silhouette Score 
z = model_2.predict(X)
print(f'Mean radius: {model_2.cluster_radius(X, z)}')
print(f'Distortion: {km.euclidean_distortion(X, z) :.3f}')
print(f'Silhouette Score: {km.euclidean_silhouette(X, z) :.3f}')

# Plot cluster assignments
C = model_2.get_centroids()
K = len(C)
_, ax = plt.subplots(figsize=(5, 5), dpi=100)
sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K), palette='tab10', data=data_2[['x0', 'x1']], ax=ax);
sns.scatterplot(x=C[:,0], y=C[:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
ax.legend().remove()


def reverse_scale(self,X):
            """
            Create data features by combining features, here: scaling values of x0 and x1

            Args:
                X (array<m,n>): a matrix of floats with
                    m rows (#samples) and n columns (#features)

            Returns:
                new_x (array<m,n>): a matrix of floats with
                    m rows (#samples) and n columns (#features)
            """
            assert X.shape[1]>1
            return np.hstack((np.expand_dims(X[:,0].copy(),axis=1),np.expand_dims(X[:,1].copy()/self.scale, axis=1)))

def change_radius(self,radius,n):
    """not used"""
    #checks if radius is too big or too small
    too_big, too_small = [], []

    for j in range (0,self.cluster):
        if radius[j]>self.radius_max:
            too_big.append(j)

        elif radius[j]<self.radius_min:
            too_small.append(j)
            rng = np.random.default_rng()
            self.centroids[j] = rng.random(n)
    #change centroids btw too big radius and too small radius
    print(too_big,too_small)

    return too_big == []

def cluster_radius(self,X,z):
    """
    not used - Computes the cluster radius

    Args:
        X (array<m,n>): a matrix of floats with
            m rows (#samples) and n columns (#features)

        z (array<m>): an array of integer with cluster assignments
            for each point.

    Returns:
        mean_radius (array<m>): an array of floats with the mean
            radius of every cluster, i.e the distance between each points and it's cluster centroid
    """
    C = self.get_centroids()
    z=z.astype(int)

    sum_radius = np.zeros(self.cluster)
    mean_radius = np.ones(self.cluster)
    effectif = np.zeros(self.cluster)

    for i in range (0,X.shape[0]-1):
        sum_radius[z[i]]+=km.euclidean_distance(X[i],C[z[i]])
        effectif[z[i]]+=1

    for k in range (self.cluster):
        if effectif[k]!=0:
            mean_radius[k]=sum_radius[k]/effectif[k]
        else:
            mean_radius[k]=0

    return mean_radius

def cluster_quality(cluster):
    """not used"""
    if len(cluster) == 0:
        return 0.0

    quality = 0.0
    for i in range(len(cluster)):
        for j in range(i, len(cluster)):
            quality += km.euclidean_distance(cluster[i], cluster[j])
    return quality / len(cluster)
