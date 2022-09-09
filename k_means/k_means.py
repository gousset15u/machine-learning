import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, cluster, dim=1, scale=1):
        self.cluster = cluster
        self.dim = dim
        self.scale = scale
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        m,n = X.shape
        centroids=np.empty([self.cluster,n])

        rng = np.random.default_rng()
        for c in range (0,self.cluster):
            centroids[c] = rng.random([n])
        self.centroids = centroids
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        
        X=self.build_scale(X)
        m,n=X.shape
        z=np.zeros(m)
        distor, distor_prev = 100, 0
        old_centroids=[]

        while (distor > 0.001 and distor_prev != distor):
            distor_prev = distor
            distor = euclidean_distortion(X,z.astype(int))

            for i in range (0,m):
                dist_eucl = np.ones(self.cluster)

                for j in range (0,self.cluster):
                    dist_eucl[j]=euclidean_distance(X[i,:],self.centroids[j])

                z[i] = np.argmin(dist_eucl)

            for j in range (0,self.cluster):
                sum_coord =np.empty(n)
                for i in range(m):
                    sum_coord = np.add(sum_coord,(z[i]==j)*X[i,:])

                effectif=0
                for i in range(m):
                    effectif+=(z[i]==j)

                if effectif!=0:
                    self.centroids[j]=sum_coord/effectif

                else: 
                    rng = np.random.default_rng()
                    self.centroids[j] = rng.random(n)

            old_centroids.append(self.centroids)

        self.old_centroids=old_centroids

        return z.astype(int)
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return np.hstack((np.expand_dims(self.centroids[:,0],axis=1),np.expand_dims(self.centroids[:,1]/self.scale, axis=1)))

    def build_scale(self,X):
        """
        Create data features by combining features, here: scaling values of x0 and x1
        Note: should be called during .predict()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            new_x (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
        """
        assert X.shape[1]>1
        return np.hstack((np.expand_dims(X[:,0].copy(),axis=1),np.expand_dims(X[:,1]*self.scale, axis=1)))

    
    
# --- Some utility functions 

def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    return np.mean((b - a) / np.maximum(a, b))
