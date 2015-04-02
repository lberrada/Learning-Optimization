# coding utf-8

import numpy as np
import pandas as pd
import sklearn.cross_validation as skc

from lq import least_squares

def ordinary_least_squares(X, Y, n_folds = 10):
    """apply ordinary-least-squares regression on (X,Y) with an interception term
    
    :param X predictors data
    :param Y result
    :param print_option 
    :return: est """
    
    # get predictors
    predictors = ["const"] + X.keys().tolist()
    
    # get data shape
    n, p = X.shape

    # create cross-validation indices
    indices = skc.KFold(n, n_folds = n_folds)
        
    # create data matrix by block
    one_vec = np.ones((n,1))
    big_X = pd.DataFrame(np.concatenate([one_vec, X], axis = 1), columns = predictors)
    
    print "No parameter to tune for OLS estimation"
    r2 = 0
    for train_indices, test_indices in indices:
            
        # create training and validation sets    
        X_train = big_X.iloc[train_indices]
        Y_train = Y.iloc[train_indices]
        X_test = big_X.iloc[test_indices]
        Y_test = Y.iloc[test_indices]

        clf = least_squares(X_train, Y_train)
        beta = clf.coef_.tolist()
        r2 += clf.score(X_test, Y_test)
        
    r2 /= n_folds

    # create estimator dictionary
    est = dict()

    # relate to predictors
    for ind in range(len(predictors)):
        est[predictors[ind]] = beta[ind]
    
    # print results
    print "="*50
    print "Cross-Validation"
    print "="*50
    print "Adjusted R2: ", r2

    print "="*50
    print "Coefficients"
    print "="*50
    for key in est.keys():
        print key + " : " + str(est[key])
    