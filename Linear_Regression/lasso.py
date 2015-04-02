# coding utf-8

import numpy as np
import sklearn.linear_model as slm

from error import get_error

def lasso_regression(X, Y, mu, print_option = True):
    """apply lasso regression on (X,Y) with L1-regularization parameter mu
    
    :param X predictors data
    :param Y result
    :param print_option 
    :return: est """
    
    # get predictors
    predictors = ["const"] + X.keys().tolist()
    
    # get data shape
    n, p = X.shape
    shape = (n, p)
    
    # create data matrix by block
    one_vec = np.ones((n,1))
    big_X = np.concatenate([one_vec, X], axis = 1)
    
    # compute regression
    clf = slm.Lasso(alpha = mu, fit_intercept = False)
    clf.fit(big_X, Y)
    
    # get estimator
    beta = clf.coef_
    
    # create estimator dictionary
    est = dict()
    
    # relate to predictors
    for ind in range(len(predictors)):
        est[predictors[ind]] = beta[ind]
        
    if print_option:
        print est
        
    # return result
    return beta, est