# coding utf-8

import scipy.linalg as slg
import numpy as np
import sklearn.cross_validation as skc
import pandas as pd

from lq import least_squares
from visualize import visualize

def estimate(X, Y, mu, d):
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
    beta, est = least_squares(big_X, big_Y, predictors)
    
    # retru result
    return beta, est

def cross_validate(X, Y, n_folds):
    """perform cross validation for lasso regression"""
    
    predictors = ["const"] + X.keys().tolist()
    
    # auxiliary variables to store best results
    best_r2 = 0
    best_mu = 0
    best_d = None
    
    # declare variables for data sizes
    n, _ = X.shape
    
    # create cross-validation indices
    indices = skc.KFold(n, n_folds = n_folds)
    
    mu_values = np.arange(0.001,10,1)
    d_values = np.arange(1,X.shape[1])
    r2_values = np.zeros(len(mu_values) * len(d_values))
    i = 0
    r2_values = np.zeros((len(mu_values),len(d_values)))
    for mu in mu_values:
        j=0
        for d in d_values:
            r2_values[i][j] = 0
            for train_indices, test_indices in indices:
                
                # create training and validation sets  
                ones_test = np.ones((len(test_indices),1))
                X_train = X.iloc[train_indices]  
                Y_train = Y.iloc[train_indices]
                X_test = X.iloc[test_indices].as_matrix()
                X_test = pd.DataFrame(np.concatenate([ones_test, X_test], axis = 1), columns = predictors)
                Y_test = Y.iloc[test_indices]
    
                # transform into dataframes to keep variables names
                beta, _ = estimate(X_train, Y_train, mu, d)
                r2_values[i][j] += beta.score(X_test, Y_test)
            
            r2_values[i][j] /= n_folds
        
            if r2_values[i][j] > best_r2:
                best_r2 = r2_values[i][j]
                best_mu = mu
                best_d = d
            j += 1
        i += 1
    
    r2 = 0    
    for train_indices, test_indices in indices:
                
        # create training and validation sets  
        ones_test = np.ones((len(test_indices),1))
        X_train = X.iloc[train_indices]  
        Y_train = Y.iloc[train_indices]
        X_test = X.iloc[test_indices].as_matrix()
        X_test = pd.DataFrame(np.concatenate([ones_test, X_test], axis = 1), columns = predictors)
        Y_test = Y.iloc[test_indices]

        # transform into dataframes to keep variables names
        beta, est = estimate(X_train, Y_train, mu, d)
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
    
    visualize("EDE", mu_values, r2_values, d_values)