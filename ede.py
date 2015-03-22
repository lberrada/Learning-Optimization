# coding utf-8

import scipy.linalg as slg
import numpy as np

from lq import least_squares

def exterior_derivative_estimation(X, Y, mu, d, print_option = True):
    """apply exterior derivatice regression on (X,Y) with regularization parameter mu and dimension parameter d
    
    :param X predictors data
    :param Y result
    :param mu L2-regularization parameter
    :param d number of dimensions kept from SVD
    :param print_option 
    :return: est"""
    
    # get data shape
    n, p = X.shape
    shape = (n, p)
    
    # get predictors
    predictors = ["const"] + X.keys().tolist()
    
    # perform singular value decomposition
    U, S, V = slg.svd(X)
    
    # keep only p * (p-d) dimensions from V (eigen values were computed in non-increasing order in svd)
    V = V.transpose()[d:]
    V = V.transpose()
    
    # compute Pi matrix
    Pi = np.dot(V, V.transpose())
    
    # find n number of rows and p number of cols
    n, p = X.shape
    
    # create numpy arrays for block matrices (X)
    X_matrix = X.as_matrix()
    one_vec = np.ones((n,1))
    mu_matrix = mu * Pi
    zero_vec = np.zeros((p,1))
    
    # concatenate in upper and lower block matrices (X)
    upper_block = np.concatenate([one_vec, X_matrix], axis = 1)
    lower_block = np.concatenate([zero_vec, mu_matrix], axis = 1)
        
    # concatenate in big matrix (X)
    big_X = np.concatenate([upper_block, lower_block], axis = 0)
        
    # create numpy arrays for block matrices (Y)
    Y_vec = Y.as_matrix()
    zero_vec = np.zeros(p)
    
    # concatenate in big vector
    big_Y = np.concatenate([Y_vec, zero_vec])
    
    # apply ordinary least squares on big matrices
    est, r2 = least_squares(big_X, big_Y, shape, predictors, print_option)
    
    # retru result
    return est, r2