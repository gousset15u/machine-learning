from re import M
import numpy as np 
import pandas as pd 
import math
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:


    def __init__(self, alpha, dim):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.alpha = alpha
        self.dim = dim

    def fit(self, X, y):
        """
        Estimates parameters for the classifier 
            with a stochastic gradient descent
        
        Args:
            X (array<m,n>): a numpy matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a numpy vector of floats containing 
                m binary 0.0/1.0 labels
        """
        if self.dim == 2:
            X=self.build_add(X)
        m,n = X.shape    
        teta=np.ones(n+1)

        # rng = np.random.default_rng()
        # for c in range (0,n+1):
        #     teta[c] = rng.random()

        for i in range (m):
            xi=np.insert(X[i,:].copy(),0,1)
            hypi=sigmoid(np.matmul(np.transpose(teta),xi))

            # teta 0
            teta[0]+=self.alpha*(y[i]-hypi)

            # teta 1 to n+1
            for j in range (1,n+1):
                teta[j]+=self.alpha*(y[i]-hypi)*xi[j]

        self.teta=teta
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        if self.dim == 2:
            X=self.build_add(X)
        m,n = X.shape
        output = np.zeros(m)

        for i in range (m):
            xi=np.insert(X[i,:].copy(),0,1)
            output[i]=sigmoid(np.matmul(np.transpose(self.teta),xi))
        return output


    def build_add(self,X):
        """
        Create data features by combining features, here: 
            adding absolute values of x0 and x1
        Note: should be called during .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            new_x (array<m,n+1>): a matrix of floats with 
                m rows (#samples) and n+1 columns (#features
        """
        assert X.shape[1]>1
        return np.hstack((np.expand_dims(abs(X[:,0]+X[:,1]), axis=1),X.copy()))

    def grid_search(sel,alpha,dim):
        """
        Update hyperparameters alpha and dim to the best accuracy possible
        
        Note: should be called before .fit()
        
        Args:
            alpha (array<n>): a vector of floats in the range [0, 1] 
                containing m potential learning rate
            dim (integer): a vector of integers in the range [0, 5] 
                containing m potential dimensions for the hypothesis function
        """

        # TODO: Implement 
        raise NotImplementedError()

# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

def convergence(teta,ref,eps):
    assert teta.shape == ref.shape

    n=teta.shape
    flag = 1
    for j in range (n[0]):
        flag=flag*(teta[j]-ref[j]>eps)
    return flag
