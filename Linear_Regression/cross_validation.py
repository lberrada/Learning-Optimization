# coding utf-8

import numpy as np
import pandas as pd

from lasso import lasso_regression
from ridge import ridge_regression
from ede import exterior_derivative_estimation
from error import get_error
from visualize import visualize

def cross_validate(method, X, Y, training_fraction=0.8):
    """perform cross validation for specified method"""
    
    print "Cross validation for " + method + " method"
    
    # auxiliary variables to store best results
    best_r2 = 0
    best_mu = 0
    best_d = None
    best_est = dict()
    
    # declare variables for data sizes
    n, p = X.shape
    n_train = int(0.8 * n)
    n_test = n - n_train
    shape = (n_test, p)
    
    # shuffle indices
    shuffled_indices = range(n)
    np.random.shuffle(shuffled_indices)
    
    # create training and validation sets    
    X_train = X.iloc[shuffled_indices[:n_train]]
    Y_train = Y.iloc[shuffled_indices[:n_train]]
    X_test = X.iloc[shuffled_indices[n_train:]]
    Y_test = Y.iloc[shuffled_indices[n_train:]]
    
    # transform into dataframes to keep variables names
    one_test = np.ones((n_test,1))
    X_train = pd.DataFrame(X_train, columns = X.keys().tolist())
    X_test = pd.DataFrame(np.concatenate([one_test, X_test], axis = 1), columns = ["const"]+X.keys().tolist())
    
    # declare additional variables fror EDE method
    if method == "EDE":
        mu_values = np.arange(0.001,0.1,0.01)
        d_values = np.arange(1,X.shape[1])
        r2_values = np.zeros(len(mu_values) * len(d_values))
    else:
        d_values = None
        mu_values = np.arange(0.000001,0.02,0.0001)
        r2_values = np.zeros(len(mu_values))
    
    ind=0
    
    # LASSO
    if method == "lasso":
        for mu in mu_values:
            beta, _ = lasso_regression(X_train, Y_train, mu, print_option = False)
            r2_values[ind] = get_error(X_test, Y_test, beta, shape)
            if r2_values[ind] > best_r2:
                best_r2 = r2_values[ind]
                best_mu = mu
                best_est = beta
            ind += 1
    
    # LEAST SQUARES    
    elif method == "least squares":
        print "No parameter to tune, no need for cross-validation"
    
    # RIDGE   
    elif method == "ridge":
        for mu in mu_values:
            beta, _ = ridge_regression(X_train, Y_train, mu, print_option = False)
            r2_values[ind] = get_error(X_test, Y_test, beta, shape)
            if r2_values[ind] > best_r2:
                best_r2 = r2_values[ind]
                best_mu = mu
                best_est = beta
            ind += 1
    
    # EDE    
    elif method == "EDE":
        i, j = 0, 0
        r2_values = np.zeros((len(mu_values),len(d_values)))
        for mu in mu_values:
            j=0
            for d in d_values:
                beta, _ = exterior_derivative_estimation(X_train, Y_train, mu, d, print_option = False)
                r2_values[i][j] = get_error(X_test, Y_test, beta, shape)
                if r2_values[i][j] > best_r2:
                    best_r2 = r2_values[i][j]
                    best_mu = mu
                    best_d = d
                    best_est = beta
                j += 1
            i += 1
        
    else:
        print "Error: method should be 'lasso', 'least squares', 'ridge' or 'EDE'"
        return
    
    # print results
    print "Best error: ", best_r2
    print "Best mu: ", best_mu
    if best_d != None:
        print "Best d: ", best_d
    
    # visualize results 
    visualize(method, mu_values, r2_values, d_values)
        
    return best_est