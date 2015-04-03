# coding utf-8

import numpy as np
import sklearn.cross_validation as skc
import pandas as pd

from lq import least_squares
from visualize import visualize

def estimate(X, Y, mu):
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
    beta, est = least_squares(big_X, big_Y, predictors)
    
    # retru result
    return beta, est

def cross_validate(X, Y, n_folds):
    """perform cross validation for ridge regression"""
    
    predictors = ["const"] + X.keys().tolist()
        
    # auxiliary variables to store best results
    best_r2 = 0
    best_mu = 0
    best_d = None
    
    # declare variables for data sizes
    n, _ = X.shape
    
    # create cross-validation indices
    indices = skc.KFold(n, n_folds = n_folds)
    
    # initialize index
    ind=0
    
    mu_values = np.arange(0.00001,5,0.05)
    r2_values = np.zeros(len(mu_values))
    for mu in mu_values:
        r2_values[ind] = 0
        for train_indices, test_indices in indices:
            
            # create training and validation sets    
            X_train = X.iloc[train_indices]
            Y_train = Y.iloc[train_indices]
            X_test = X.iloc[test_indices]
            Y_test = Y.iloc[test_indices]

            # transform into dataframes to keep variables names
            one_test = np.ones((len(test_indices),1))
            X_test = pd.DataFrame(np.concatenate([one_test, X_test], axis = 1), columns = predictors)
            beta, _ = estimate(X_train, Y_train, mu)
            r2_values[ind] += beta.score(X_test, Y_test)
        
        r2_values[ind] /= n_folds
        if r2_values[ind] > best_r2:
            best_r2 = r2_values[ind]
            best_mu = mu
        ind += 1
    
    r2 = 0    
    for train_indices, test_indices in indices:
            
        # create training and validation sets    
        X_train = X.iloc[train_indices]
        Y_train = Y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        Y_test = Y.iloc[test_indices]

        # transform into dataframes to keep variables names
        one_test = np.ones((len(test_indices),1))
        X_test = pd.DataFrame(np.concatenate([one_test, X_test], axis = 1), columns = predictors)
        beta, est = estimate(X_train, Y_train, mu)
        r2 += beta.score(X_test, Y_test)
        
    r2 /= n_folds
        
    # print results
    print "="*50
    print "Cross-Validation"
    print "="*50
    print "Adjusted R2: ", r2
    print "Best mu: ", best_mu
    if best_d != None:
        print "Best d: ", best_d

    print "="*50
    print "Coefficients"
    print "="*50
    for key in est.keys():
        print key + " : " + str(est[key])
    
    visualize("ridge", mu_values, r2_values, None)