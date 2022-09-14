import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, cluster, dim=1):
        self.cluster = cluster
        self.dim = dim
        self.radius_min = 0.44
        self.radius_max = 0.51
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        m,n = X.shape
        centroids=np.empty([self.cluster,n])

        #initalize centroids to random points
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
        self.old_centroids=[]
        prev_z=np.ones(m)

        while (not np.array_equal(z,prev_z)):
            prev_z = z.copy()

            #assign each point to the nearest centroid
            for i in range (0,m):
                dist_eucl = np.ones(self.cluster)

                for j in range (0,self.cluster):
                    dist_eucl[j]=euclidean_distance(X[i,:],self.centroids[j]) 

                z[i] = np.argmin(dist_eucl)

            #update centroids to the average location of its assigned points
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

            #add centroids to record
            self.old_centroids.append(self.centroids.copy())

        return z.astype(int)

    def check_radius(self,X,z,n):
        #checks that radius isn't too big or too small
        radius = self.cluster_radius(X,z)
        too_big, too_small = [], []

        for j in range (0,self.cluster):
            if radius[j]>self.radius_max:
                too_big.append(j)

            elif radius[j]<self.radius_min:
                too_small.append(j)
        return too_big,too_small

    def centroids_radius(self,list):
        #change centroids btw too big radius and too small radius
        #lol
        return None


    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        """
        return np.hstack((np.expand_dims(self.centroids[:,0],axis=1),np.expand_dims(self.centroids[:,1]/self.scale, axis=1)))
        
    def get_init_centroids(self):
        """
        Returns the centroids randomly initialized by the K-mean algorithm
        """
        return np.hstack((np.expand_dims(self.old_centroids[0][:,0],axis=1),np.expand_dims(self.old_centroids[0][:,1]/self.scale, axis=1)))


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
        self.scale = (np.amax(X[:,0])-np.amin(X[:,0]))/(np.amax(X[:,1])-np.amin(X[:,1]))
        return np.hstack((np.expand_dims(X[:,0].copy(),axis=1),np.expand_dims(X[:,1]*self.scale, axis=1)))

    def cluster_radius(self,X,z):
        """
        Computes the cluster radius
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)

            z (array<m>): an array of integer with cluster assignments
                for each point. 
            
        Returns: 
            mean_radius (array<m>): an array of floats with the mean 
                radius of every cluster, i.e the distance between each points and it's cluster centroid
        """
        m,n = X.shape
        C = self.get_centroids()
        z=z.astype(int)

        sum_radius = np.zeros(self.cluster)
        mean_radius = np.ones(self.cluster)
        effectif = np.zeros(self.cluster)

        for i in range (0,m-1):
            sum_radius[z[i]]+=euclidean_distance(X[i],C[z[i]])
            effectif[z[i]]+=1

        for k in range (self.cluster):
            if effectif[k]!=0:
                mean_radius[k]=sum_radius[k]/effectif[k]
            else:
                mean_radius[k]=0

        return mean_radius
        
def cluster_quality(cluster):
    if len(cluster) == 0:
        return 0.0

    quality = 0.0
    for i in range(len(cluster)):
        for j in range(i, len(cluster)):
            quality += euclidean_distance(cluster[i], cluster[j])
    return quality / len(cluster)

    
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
