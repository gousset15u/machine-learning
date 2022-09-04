from inspect import ArgSpec
import numpy as np 
import pandas as pd 
import random as rd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, cluster):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.cluster = cluster
        self.eps = 0.001
        self.list_z=[]
        self.list_centroid=[]
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # TODO: Implement
        m,n = X.shape
        centroids=np.empty([self.cluster,n])

        rng = np.random.default_rng()

        for c in range (0,self.cluster):
            centroids[c] = rng.random([n])
        # if self.cluster==10:
        #     centroids[0]=[0.9,1]
        #     centroids[1]=[0.7,1]
        #     centroids[2]=[0.7,3]
        #     centroids[3]=[0.9,5]
        #     centroids[4]=[0.6,7]
        #     centroids[5]=[0.5,7]
        #     centroids[6]=[0.8,10]
        #     centroids[7]=[0.2,4]
        #     centroids[8]=[0,1]
        #     centroids[9]=[0.1,6]
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
        m,n=X.shape
        z=np.empty(m)
        distor = 100
        distor_prev = 0

        list_z=[]
        list_centroid=[]
        compt = 0
        
        # x0 = np.stack((X[:,0],np.zeros(m)),axis=-1)
        # x1 = np.stack((X[:,1],np.zeros(m)),axis=-1)

        while (distor > self.eps and distor_prev != distor):
            compt+=1
            distor_prev = distor
            distor = euclidean_distortion(X,z.astype(int))

            for i in range (0,m):
                dist_eucl = np.ones(self.cluster)
                # dist_eucl_x0 = np.ones(self.cluster)
                # dist_eucl_x1 = np.ones(self.cluster)

                for j in range (0,self.cluster):
                    dist_eucl[j]=euclidean_distance(X[i,:],self.centroids[j])
                    # dist_eucl_x0[j]=euclidean_distance(x0[i,:],self.centroids[j])
                    # dist_eucl_x1[j]=euclidean_distance(x1[i,:],self.centroids[j])
                # dist_eucl = euclidean_distance(X[i,:],self.centroids)
                # dist_eucl = cross_euclidean_distance(X[i,:],self.centroids)

                # if compt%3==0:
                #     z[i] = np.argmin(dist_eucl)
                # elif compt%2==1:
                #     z[i] = np.argmin(dist_eucl_x0)
                # else:
                #     z[i] = np.argmin(dist_eucl_x1)
                z[i] = np.argmin(dist_eucl)
                #z[i] = argmin(X[i,:],self.centroids,self.cluster)

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
                    print("zero")
                    #TODO: find better method to delete a centroid
                    rng = np.random.default_rng()
                    self.centroids[j] = rng.random(n)
            list_z.append(z)
            list_centroid.append(self.centroids)

        self.list_z=list_z
        self.list_centroid=list_centroid

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
        return self.centroids

    def get_list_centroids(self):
        return self.list_centroid

    def get_list_z(self):
        return self.list_z
    
    
# --- Some utility functions 

def argmin(x,mu,c):
    """
    NOT USED - Finds the closest cluster centroid for the training example x
        using the Enclidian distance between two set of points
    
    Args:
        x (array<n>): n-length vector with datapoints 
        mu (array<n>): n-length vector of a cluster's centroid
        c (integer): integer of number of clusters
    
    Returns:
        An integer with the closest cluster centroid
    """
    min=euclidean_distance(x,mu[0])
    cent=0
    for j in range (0,c):
        arg=euclidean_distance(x,mu[j])
        # TODO: why is it the same euclidian distance btw different xi values with same mu[j] ? 
        if arg<min:
            min=arg
            cent=j
    return cent


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
