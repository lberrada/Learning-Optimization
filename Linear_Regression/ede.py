# coding utf-8

import scipy.linalg as slg
import numpy as np
import pandas as pd
import sklearn.cross_validation as skc

from lq import least_squares
from visualize import visualize

def exterior_derivative_estimation(X, Y, n_folds = 10):
    """apply exterior derivatice regression on (X,Y) with regularization parameter mu and dimension parameter d
    
    :param X predictors data
    :param Y result"""
    
    # get predictors
    predictors = ["const"] + X.keys().tolist()
    
    # perform singular value decomposition
    U, S, V = slg.svd(X)
    
    # find n number of rows and p number of cols
    n, p = X.shape
        
    # create numpy arrays for block matrices (Y)
    Y_vec = Y.as_matrix()
    zero_vec1 = np.zeros(p)
    
    # concatenate in big vector
    big_Y = pd.DataFrame(np.concatenate([Y_vec, zero_vec1]))
    
    # auxiliary variables to store best results
    best_r2 = 0
    best_mu = 0
    best_d = None
    
    # create cross-validation indices
    indices = skc.KFold(n, n_folds = n_folds)
    
    # initialize index
    ind=0
    
    X_matrix = X.as_matrix()
    one_vec = np.ones((n,1))
    zero_vec2 = np.zeros((p,1))
    
    # create data matrix by block
    one_vec = np.ones((n,1))
    mu_values = np.arange(0.00001,10,0.1)
    d_values = np.arange(1, X.shape[1])
    r2_values = np.zeros((len(d_values),len(mu_values)))
    i = 0
    for d in d_values:
        # keep only p * (p-d) dimensions from V (eigen values were computed in non-increasing order in svd)
        V = V.transpose()[d:]
        V = V.transpose()
        
        # compute Pi matrix
        Pi = np.dot(V, V.transpose())
        for mu in mu_values:
            j=0

            mu_matrix = mu * Pi
            
            # concatenate in upper and lower block matrices (X)
            lower_block = np.concatenate([zero_vec2, mu_matrix], axis = 1)
            
            r2_values[i][j] = 0
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
                r2_values[i][j] += clf.score(X_test, Y_test)
            
            # normalize score
            r2_values[i][j] /= n_folds
                         
            if r2_values[i][j] > best_r2:
                best_r2 = r2_values[i][j]
                best_mu = mu
                best_d = d
            j += 1
        i += 1     
        
    V = V.transpose()[best_d:]
    V = V.transpose()
    
    # compute Pi matrix
    Pi = np.dot(V, V.transpose())
    mu_matrix = best_mu * Pi
        
    # concatenate in upper and lower block matrices (X)
    upper_block = np.concatenate([one_vec, X_matrix], axis = 1)
    lower_block = np.concatenate([zero_vec2, mu_matrix], axis = 1)
    
    # concatenate in big matrix (X)
    big_X = pd.DataFrame(np.concatenate([upper_block, lower_block], axis =0), columns = predictors) 
    
    clf = least_squares(big_X, big_Y)
    beta = clf.coef_.tolist()[0]
    r2 = clf.score(big_X, big_Y)   
     
    
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
        
    visualize("EDE", mu_values, r2_values, d_values)