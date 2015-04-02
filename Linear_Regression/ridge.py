# coding utf-8

import numpy as np
import pandas as pd
import sklearn.cross_validation as skc
from visualize import visualize

from lq import least_squares

def ridge_regression(X, Y, n_folds = 10):
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
    
    # get predictors
    predictors = ["const"] + X.keys().tolist()
    
    # auxiliary variables to store best results
    best_r2 = 0
    best_mu = 0
    best_d = None
    
    # create cross-validation indices
    indices = skc.KFold(n, n_folds = n_folds)
    
    # initialize index
    
    # create numpy arrays for block matrices (Y)
    Y_vec = Y.as_matrix()
    zero_vec1 = np.zeros(p)
    
    # concatenate in big vector
    big_Y = pd.DataFrame(np.concatenate([Y_vec, zero_vec1]))
    
    one_vec = np.ones((n,1))
    zero_vec2 = np.zeros((p,1))
    
    ind=0
    
    # create data matrix by block
    one_vec = np.ones((n,1))
    mu_values = np.arange(0.00001,10,0.1)
    r2_values = np.zeros(len(mu_values))
    for mu in mu_values:
        # create numpy arrays for block matrices (X)
        mu_matrix = mu * np.identity(p)
        
        # concatenate in upper and lower block matrices (X)
        lower_block = np.concatenate([zero_vec2, mu_matrix], axis = 1)
        
        r2_values[ind] = 0
        for train_indices, test_indices in indices:
            
            # concatenate in upper and lower block matrices (X)
            upper_block_train = np.concatenate([one_vec[train_indices], X.iloc[train_indices].as_matrix()], axis = 1)
            upper_block_test = np.concatenate([one_vec[test_indices], X.iloc[test_indices].as_matrix()], axis = 1)
            
            # concatenate in big matrix (X)
            X_train = pd.DataFrame(np.concatenate([upper_block_train, lower_block], axis =0), columns = predictors)
            X_test = pd.DataFrame(np.concatenate([upper_block_test, lower_block], axis =0), columns = predictors)
            
            # create validation sets   
            Y_train = pd.DataFrame(np.concatenate([Y.iloc[train_indices].as_matrix(), zero_vec1], axis =0)) 
            Y_test = pd.DataFrame(np.concatenate([Y.iloc[test_indices].as_matrix(), zero_vec1], axis =0)) 
            
            # fit regression and compute score
            clf = least_squares(X_train, Y_train)
            r2_values[ind] += clf.score(X_test, Y_test)
        
        # normalize score
        r2_values[ind] /= n_folds
        if r2_values[ind] > best_r2:
            best_r2 = r2_values[ind]
            best_mu = mu
        ind += 1      
    
    # create numpy arrays for block matrices (X)
    X_matrix = X.as_matrix()
    one_vec = np.ones((n,1))
    mu_matrix = best_mu * np.identity(p)
    zero_vec = np.zeros((p,1))
    
    # concatenate in upper and lower block matrices (X)
    upper_block = np.concatenate([one_vec, X_matrix], axis = 1)
    lower_block = np.concatenate([zero_vec, mu_matrix], axis = 1)
    
    # concatenate in big matrix (X)
    big_X = pd.DataFrame(np.concatenate([upper_block, lower_block], axis =0), columns = predictors)
    
    clf = least_squares(big_X, big_Y)
    r2 = clf.score(X_test, Y_test)   
    beta = clf.coef_.tolist()[0]
    
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
    print "Best mu: ", best_mu
    if best_d != None:
        print "Best d: ", best_d

    print "="*50
    print "Coefficients"
    print "="*50
    for key in est.keys():
        print key + " : " + str(est[key])
    
    visualize("ridge", mu_values, r2_values, None)
        