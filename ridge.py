# coding utf-8

import numpy as np

from lq import least_squares

def ridge_regression(X, Y, mu, print_option = True):
    """apply ridge regression on (X,Y) with regularization parameter mu
    
    :param X predictors data
    :param Y result
    :param mu L2-regularization parameter
    :param print_option 
    :return: est"""
    
    # get predictors
    predictors = ["const"] + X.keys().tolist()
    
    # find n number of rows and p number of cols
    n, p = X.shape
    shape = (n, p)
    
    # create numpy arrays for block matrices (X)
    X_matrix = X.as_matrix()
    one_vec = np.ones((n,1))
    mu_matrix = mu * np.identity(p)
    zero_vec = np.zeros((p,1))
    
    # concatenate in upper and lower block matrices (X)
    upper_block = np.concatenate([one_vec, X_matrix], axis = 1)
    lower_block = np.concatenate([zero_vec, mu_matrix], axis = 1)
    
    # concatenate in big matrix (X)
    big_X = np.concatenate([upper_block, lower_block], axis =0)
        
    # create numpy arrays for block matrices (Y)
    Y_vec = Y.as_matrix()
    zero_vec = np.zeros(p)
    
    # concatenate in big vector
    big_Y = np.concatenate([Y_vec, zero_vec])
    
    # apply ordinary least squares on big matrices
    beta, est, r2 = least_squares(big_X, big_Y, shape, predictors, print_option)
    
    # retru result
    return beta, est, r2