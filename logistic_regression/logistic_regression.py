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
        X=self.build_dim(X)
        m,n = X.shape    
        teta=np.ones(n+1)

        for i in range (m):
            xi=np.insert(X[i,:].copy(),0,1)
            hypi=sigmoid(np.matmul(np.transpose(teta),xi))

            # teta 0
            teta[0]+=self.alpha*(y[i]-hypi)

            # teta 1 to n+1
            for j in range (1,n+1):
                teta[j]+=self.alpha*(y[i]-hypi)*xi[j]

        self.teta=teta

    def fit_batch(self, X, y):
        """
        NOT USED - Estimates parameters for the classifier
            with a batch gradient descent
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        m,n = X.shape
        teta=np.ones(n+1)

        for compt in range (200):
            # switch with a check for convergence

            # teta 0
            sum0=0
            for i in range (m-1):
            #xi=np.insert(X.iloc[i,:].values,0,1)
                xi=np.insert(X[i,:],0,1)
                hypi=sigmoid(np.matmul(np.transpose(teta),xi))
                sum0+=(y[i]-hypi)

            teta[0]+=self.alpha*sum0

            # teta 1 to n+1
            for j in range (1,n+1):
                
                sumj=0
                for i in range (m-1):
                    xi=np.insert(X[i,:],0,1)
                    hypi=sigmoid(np.matmul(np.transpose(teta),xi))
                    sumj+=(y[i]-hypi)*X[i,j-1]

                teta[j]+=self.alpha*sumj

        self.teta=teta
        print(f"teta after fit:\n{self.teta}\n")

    
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
        X=self.build_dim(X)
        m,n = X.shape
        output = np.zeros(m)

        for i in range (m):
            xi=np.insert(X[i,:].copy(),0,1)
            output[i]=sigmoid(np.matmul(np.transpose(self.teta),xi))
        return output

        
    def build_dim(self,X):
        """
        Create data features using different dimension
        Note: should be called during .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and 2 columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        m,n = X.shape

        if self.dim==1:
            return X
        
        else:
            sign_dim = int(math.copysign(1,self.dim))
            x_dim = n*self.dim if sign_dim else n*(abs(self.dim)+1)
            beg = 2 if sign_dim else -1
            new_x = np.empty([m,x_dim])

            for i in range (m):
                xi = X[i,:].copy()

                for d in range (beg,self.dim+sign_dim,sign_dim):
                    xi_d=np.power(xi,d)
                    xi = np.hstack((xi,xi_d))
                new_x[i] = xi
            
            return new_x
        
    def build_dim_col(self,X):
        """
        Create data features using different dimension (construction by columns)
        Note: should be called during .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and 2 columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        m,n = X.shape

        if self.dim==1:
            return X
        
        elif self.dim==1:
            sign_dim = int(math.copysign(1,self.dim))
            x_dim = n*self.dim if sign_dim else n*(abs(self.dim)+1)
            beg = 2 if sign_dim else -1
            new_x = np.empty([m,x_dim])

            for i in range (m):
                xi = X[i,:].copy()

                for d in range (beg,self.dim+sign_dim,sign_dim):
                    xi_d=np.power(xi,d)
                    xi = np.hstack((xi,xi_d))
                new_x[i] = xi
            
            return new_x

        else:
            new_x = X.copy()

            for d in range (beg,self.dim+sign_dim,sign_dim):
                col_d=np.power(xi,d)
                col_d=np.hstack((xi,xi_d))
            new_x[i] = xi

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
