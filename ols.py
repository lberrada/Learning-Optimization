# coding utf-8

import numpy as np

from lq import least_squares

def ordinary_least_squares(X, Y, print_option = True):
    """apply ordinary-least-squares regression on (X,Y) with an interception term
    
    :param X predictors data
    :param Y result
    :param print_option 
    :return: est """
    
    # get predictors
    predictors = ["const"] + X.keys().tolist()
    
    # get data shape
    n, p = X.shape
    shape = (n,p)
    
    # create data matrix by block
    one_vec = np.ones((n,1))
    big_X = np.concatenate([one_vec, X], axis = 1)
    
    # compute estimator
    beta, est = least_squares(big_X, Y, shape, predictors, print_option)
    
    # return result
    return beta, est