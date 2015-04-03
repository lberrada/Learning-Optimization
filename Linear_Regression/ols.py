# coding utf-8

import numpy as np
import sklearn.cross_validation as skc
import pandas as pd

from lq import least_squares
from visualize import visualize


def estimate(X, Y, *args, **kwargs):
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
    beta, est = least_squares(big_X, Y, predictors)
    r2 = beta.score(big_X, Y)
    
    # print results
    print "="*50
    print "Error Estimation"
    print "="*50
    print "Adjusted R2: ", r2

    print "="*50
    print "Coefficients"
    print "="*50
    for key in est.keys():
        print key + " : " + str(est[key])
    
    # return result
    return beta, est