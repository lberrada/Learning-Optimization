# coding utf-8

import numpy as np
import pandas as pd
import sklearn.cross_validation as skc

from lasso import lasso_regression
from ridge import ridge_regression
from ede import exterior_derivative_estimation
from ols import ordinary_least_squares
from error import get_error
from visualize import visualize

def cross_validate(method, X, Y, n_folds=10):
    """perform cross validation for specified method"""
    
    print "Cross validation for " + method + " method"
    
    # auxiliary variables to store best results
    best_r2 = 0
    best_mu = 0
    best_d = None
    best_est = dict()
    
    # declare variables for data sizes
    n, p = X.shape
    
    # create cross-validation indices
    indices = skc.KFold(n, n_folds = n_folds)
    
    # initialize index
    ind=0
    
    # LASSO
    if method == "lasso":
        d_values = None
        mu_values = np.arange(0.00001,0.05,0.001)
        r2_values = np.zeros(len(mu_values))
        for mu in mu_values:
            r2_values[ind] = 0
            for train_indices, test_indices in indices:
                
                shape = (len(test_indices), p)
    
                # create training and validation sets    
                X_train = X.iloc[train_indices]
                Y_train = Y.iloc[train_indices]
                X_test = X.iloc[test_indices]
                Y_test = Y.iloc[test_indices]
    
                # transform into dataframes to keep variables names
                one_test = np.ones((len(test_indices),1))
                X_train = pd.DataFrame(X_train, columns = X.keys().tolist())
                X_test = pd.DataFrame(np.concatenate([one_test, X_test], axis = 1), columns = ["const"]+X.keys().tolist())
                X_test = pd.DataFrame(X_test, columns = X.keys().tolist())
                beta, est = lasso_regression(X_train, Y_train, mu, print_option = False)
#                 r2_values[ind] += get_error(X_test, Y_test, beta, shape)
                r2_values[ind] += beta.score(X_test, Y_test)
            
            r2_values[ind] /= n_folds
            if r2_values[ind] > best_r2:
                best_r2 = r2_values[ind]
                best_mu = mu
                best_est = est
            ind += 1
    
    # LEAST SQUARES    
    elif method == "least squares":
        print "No parameter to tune for least squares estimation"
        r2 = 0
        for train_indices, test_indices in indices:
                
            shape = (len(test_indices), p)

            # create training and validation sets    
            X_train = X.iloc[train_indices]
            Y_train = Y.iloc[train_indices]
            X_test = X.iloc[test_indices]
            Y_test = Y.iloc[test_indices]

            # transform into dataframes to keep variables names
            one_test = np.ones((len(test_indices),1))
            X_train = pd.DataFrame(X_train, columns = X.keys().tolist())
            X_test = pd.DataFrame(np.concatenate([one_test, X_test], axis = 1), columns = ["const"]+X.keys().tolist())
            beta, est = ordinary_least_squares(X_train, Y_train, print_option = False)
            r2 += get_error(X_test, Y_test, beta, shape)
        
        r2 /= n_folds
        best_est = est
        best_r2 = r2
        best_mu = None
        
    
    # RIDGE   
    elif method == "ridge":
        d_values = None
        mu_values = np.arange(0.001,5,0.01)
        r2_values = np.zeros(len(mu_values))
        for mu in mu_values:
            r2_values[ind] = 0
            for train_indices, test_indices in indices:
                
                shape = (len(test_indices), p)
    
                # create training and validation sets    
                X_train = X.iloc[train_indices]
                Y_train = Y.iloc[train_indices]
                X_test = X.iloc[test_indices]
                Y_test = Y.iloc[test_indices]
    
                # transform into dataframes to keep variables names
                one_test = np.ones((len(test_indices),1))
                X_train = pd.DataFrame(X_train, columns = X.keys().tolist())
                X_test = pd.DataFrame(np.concatenate([one_test, X_test], axis = 1), columns = ["const"]+X.keys().tolist())
                X_test = pd.DataFrame(X_test, columns = X.keys().tolist())
                beta, est = ridge_regression(X_train, Y_train, mu, print_option = False)
#                 r2_values[ind] += get_error(X_test, Y_test, beta, shape)
                r2_values[ind] += beta.score(X_test, Y_test)
            
            r2_values[ind] /= n_folds
            if r2_values[ind] > best_r2:
                best_r2 = r2_values[ind]
                best_mu = mu
                best_est = est
            ind += 1
    
    # EDE    
    elif method == "EDE":
        mu_values = np.arange(0.001,10,1)
        d_values = np.arange(1,X.shape[1])
        r2_values = np.zeros(len(mu_values) * len(d_values))
        i, j = 0, 0
        r2_values = np.zeros((len(mu_values),len(d_values)))
        for mu in mu_values:
            j=0
            for d in d_values:
                r2_values[i][j] = 0
                for train_indices, test_indices in indices:
                    
                    shape = (len(test_indices), p)
        
                    # create training and validation sets    
                    X_train = X.iloc[train_indices]
                    Y_train = Y.iloc[train_indices]
                    X_test = X.iloc[test_indices]
                    Y_test = Y.iloc[test_indices]
        
                    # transform into dataframes to keep variables names
                    one_test = np.ones((len(test_indices),1))
                    X_train = pd.DataFrame(X_train, columns = X.keys().tolist())
                    X_test = pd.DataFrame(np.concatenate([one_test, X_test], axis = 1), columns = ["const"]+X.keys().tolist())
                    beta, est = exterior_derivative_estimation(X_train, Y_train, mu, d, print_option = False)
                    r2_values[i][j] += get_error(X_test, Y_test, beta, shape)
                
                r2_values[i][j] /= n_folds
            
                if r2_values[i][j] > best_r2:
                    best_r2 = r2_values[i][j]
                    best_mu = mu
                    best_d = d
                    best_est = est
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
    if best_mu != None:
        visualize(method, mu_values, r2_values, d_values)
        
    return best_est