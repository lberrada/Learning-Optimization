# coding utf-8

import numpy as np
import sklearn.linear_model as slm
import sklearn.cross_validation as skc
import pandas as pd

from visualize import visualize

def estimate(X, Y, mu):
    """apply lasso regression on (X,Y) with L1-regularization parameter mu
    
    :param X predictors data
    :param Y result
    :param print_option 
    :return: est """
    
    # get predictors
    predictors = ["const"] + X.keys().tolist()
    
    # get data shape
    n, p = X.shape
    
    # create data matrix by block
    big_X = X
    
    # compute regression
    clf = slm.Lasso(alpha = mu, fit_intercept = True)
    clf.fit(big_X, Y)
    
    # get estimator
    beta = np.array([clf.intercept_] + clf.coef_.tolist())
    
    # create estimator dictionary
    est = dict()
    
    # relate to predictors
    for ind in range(len(predictors)):
        est[predictors[ind]] = beta[ind]
        
    # return result
    return clf, est

def cross_validate(X, Y, n_folds):
    """perform cross validation for lasso regression"""
    
    # get predictors
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
    
    mu_values = np.arange(0.00001,0.002,0.00001)
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
            X_train = pd.DataFrame(X_train, columns = X.keys().tolist())
            X_test = pd.DataFrame(np.concatenate([one_test, X_test], axis = 1), columns = predictors)
            X_test = pd.DataFrame(X_test, columns = X.keys().tolist())
            beta, _ = estimate(X_train, Y_train, mu)
            r2_values[ind] += beta.score(X_test, Y_test)
        
        r2_values[ind] /= n_folds
        if r2_values[ind] > best_r2:
            best_r2 = r2_values[ind]
            best_mu = mu
        ind += 1
        
    beta, est = estimate(X_train, Y_train, best_mu)
    r2 = beta.score(X_test, Y_test)
        
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
    
    visualize("lasso", mu_values, r2_values, None)
    
    
    
    
    
    
    
    
    
    
    