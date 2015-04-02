# coding utf-8

import numpy as np
import numpy.linalg as nlg

from error import get_error

def least_squares(X, Y, shape, predictors, print_option = True):
    """apply ordinary-least-squares regression on (X,Y)
    
    :param X predictors data
    :param Y result
    :param print_option 
    :return: est (dictionary)"""

    # compute X'X
    XtX = np.matrix(np.dot(X.transpose(),X))
    
    # compute beta
    beta = np.array(np.dot(np.dot(nlg.inv(XtX),X.transpose()),Y)).flatten()
        
    # create estimator dictionary
    est = dict()
    
    # relate to predictors
    for ind in range(len(predictors)):
        est[predictors[ind]] = beta[ind]
        
    # print results if needed
    if print_option:
        print est
    
    # return result
    return beta, est