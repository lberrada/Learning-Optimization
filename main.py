# coding utf-8

import pandas.io as pio
import numpy.linalg as nlg
import scipy.linalg as slg
import statsmodels.api as sap
import pandas as pd
import numpy as np
import sklearn.linear_model as slm

filename = "winequality-red.csv"

def getData(filename):
    """get data from csv file
    
    :param filename csv file name
    :return: X,Y data predictors, results (pandas data frames)"""
    
    df = pio.parsers.read_csv(filename, sep=";")
    
    # get predictors
    predictors = df.keys().tolist()
    predictors.remove("quality")
    
    # separate training predictors from result
    X = df[[key for key in predictors]]
    Y = df["quality"]
    
#     # add constant to X
#     X = sap.add_constant(X)
    return X, Y

def least_squares(X, Y, predictors, print_option = True):
    """apply ordinary-least-squares regression on (X,Y)
    
    :param X predictors data
    :param Y result
    :param print_option 
    :return: est (dictionary)"""

    # compute X'X
    XtX = np.matrix(np.dot(X.transpose(),X))
    
    # compute beta
    beta = np.array(np.dot(np.dot(nlg.inv(XtX),X.transpose()),Y)).flatten()
    
    # create estimator dictionary
    est = dict()
    
    # relate to predictors
    for ind in range(len(predictors)):
        est[predictors[ind]] = beta[ind]
        
    # print results if needed
    if print_option:
        print est
    
    # return result
    return est

def ordinary_least_squares(X, Y, print_option = True):
    """apply ordinary-least-squares regression on (X,Y) with an interception term
    
    :param X predictors data
    :param Y result
    :param print_option 
    :return: est """
    
    # get predictors
    predictors = ["const"] + X.keys().tolist()
    
    # get data shape
    n, p = X.shape
    
    # create data matrix by block
    one_vec = np.ones((n,1))
    big_X = np.concatenate([one_vec, X], axis = 1)
    
    # compute estimator
    est = least_squares(big_X, Y, predictors, print_option)
    
    # return result
    return est
    

def ridge_regression(X, Y, mu, print_option = True):
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
    est = least_squares(big_X, big_Y, predictors, print_option)
    
    # retru result
    return est

def lasso_regression(X, Y, mu, print_option = True):
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
    one_vec = np.ones((n,1))
    big_X = np.concatenate([one_vec, X], axis = 1)
    
    # compute regression
    clf = slm.Lasso(alpha = mu)
    clf.fit(big_X,Y)
    
    # get estimator
    beta = clf.coef_
    
    # create estimator dictionary
    est = dict()
    
    # relate to predictors
    for ind in range(len(predictors)):
        est[predictors[ind]] = beta[ind]
        
    if print_option:
        print est
        
    # return result
    return est
    

def exterior_derivative_estimation(X, Y, mu, d, print_option = True):
    """apply exterior derivatice regression on (X,Y) with regularization parameter mu and dimension parameter d
    
    :param X predictors data
    :param Y result
    :param mu L2-regularization parameter
    :param d number of dimensions kept from SVD
    :param print_option 
    :return: est"""
    
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
    est = least_squares(big_X, big_Y, predictors, print_option)
    
    # retru result
    return est

def estimate(method, X, Y, mu = None, d = None, print_option = True):
    
    print "Linear estimation with " + method + " method"
    
    if method == "lasso":
        est = lasso_regression(X, Y, mu, print_option)
        
    elif method == "least squares":
        est = ordinary_least_squares(X, Y, print_option)
        
    elif method == "ridge":
        est = ridge_regression(X, Y, mu, print_option)
        
    elif method == "EDE":
        est = exterior_derivative_estimation(X, Y, mu, d, print_option)
        
    else:
        print "Error: method should be 'lasso', 'least squares', 'ridge' or 'EDE'"
        return
    
    return est
    

X, Y = getData(filename)
estimate('least squares', X, Y, 1.5 , 2)

