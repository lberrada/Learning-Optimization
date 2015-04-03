# coding utf-8

import numpy as np
import sklearn.cross_validation as skc
import pandas as pd

from lq import least_squares
from visualize import visualize


def estimate(X, Y, n_folds = 10):
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
    indices = skc.KFold(n, n_folds = n_folds)
    
    r2 = 0
    for train_indices, test_indices in indices:
            
        # create training and validation sets    
        X_train = X.iloc[train_indices]
        Y_train = Y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        Y_test = Y.iloc[test_indices]

        # transform into dataframes to keep variables names
        one_train = np.ones((len(train_indices),1))
        one_test = np.ones((len(test_indices),1))
        X_train = pd.DataFrame(np.concatenate([one_train, X_train], axis = 1), columns = predictors)
        X_test = pd.DataFrame(np.concatenate([one_test, X_test], axis = 1), columns = predictors)
        beta, est = least_squares(X_train, Y_train, predictors)
        r2 += beta.score(X_test, Y_test)
        
    r2 /= n_folds
    
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