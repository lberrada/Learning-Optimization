# coding utf-8

import numpy as np
import sklearn.linear_model as slm
import sklearn.cross_validation as skc
import pandas as pd

from visualize import visualize

def lasso_regression(X, Y, n_folds = 10):
    """apply lasso regression on (X,Y)
    
    :param X predictors data
    :param Y result"""
    
    # get predictors
    predictors = ["const"] + X.keys().tolist()
    
    # get data shape
    n, p = X.shape
    
    # auxiliary variables to store best results
    best_r2 = 0
    best_mu = 0
    best_d = None
    
    # create cross-validation indices
    indices = skc.KFold(n, n_folds = n_folds)
    
    # initialize index
    ind=0
    
    # create data matrix by block
    one_vec = np.ones((n,1))
    big_X = pd.DataFrame(np.concatenate([one_vec, X], axis = 1), columns = predictors)
    d_values = None
    mu_values = np.arange(0.00001,0.05,0.001)
    r2_values = np.zeros(len(mu_values))
    for mu in mu_values:
        r2_values[ind] = 0
        for train_indices, test_indices in indices:
        
            # create training and validation sets   
            X_train = pd.DataFrame(np.concatenate([one_vec[train_indices], X.iloc[train_indices].as_matrix()], axis = 1), columns = predictors) 
            Y_train = Y.iloc[train_indices]
            X_test = pd.DataFrame(np.concatenate([one_vec[test_indices], X.iloc[test_indices].as_matrix()], axis = 1), columns = predictors) 
            Y_test = Y.iloc[test_indices]

            clf = slm.Lasso(alpha = mu, fit_intercept = False)
            clf.fit(X_train, Y_train)
            r2_values[ind] += clf.score(X_test, Y_test)
        
        r2_values[ind] /= n_folds
        if r2_values[ind] > best_r2:
            best_r2 = r2_values[ind]
            best_mu = mu
        ind += 1    
        
    clf = slm.Lasso(alpha = best_mu, fit_intercept = False)
    clf.fit(big_X, Y)
    beta = clf.coef_  
    r2 = clf.score(big_X, Y)
    
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
    
    # visualize results 
    if best_mu != None:
        visualize("lasso", mu_values, r2_values, d_values)
        