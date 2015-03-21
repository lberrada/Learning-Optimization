# coding utf-8

import pandas.io as pio
import numpy.linalg as nlg
import scipy.linalg as slg
import statsmodels.api as sap
import pandas as pd
import numpy as np

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

def ordinary_least_squares(X,Y, print_option = True):
    """apply ordinary-least-squares regression on (X,Y)
    
    :param X predictors data
    :param Y result
    :param print_option 
    :return: estimator"""
    
#     ols = sap.OLS(Y, X)
#     estimator = ols.fit()
#     if print_option:
#         print(estimator.summary())
    
    XtX = np.matrix(np.dot(X.transpose(),X))
    beta = np.dot(np.dot(nlg.inv(XtX),X.transpose()),Y)
    if print_option:
        print beta.shape
        print beta
    
    return beta

def ordinary_least_squares_with_constant(X, Y, print_option = True):
    """apply ordinary-least-squares regression on (X,Y) with an interception term
    
    :param X predictors data
    :param Y result
    :param print_option 
    :return: estimator """
    
    # get data shape
    n, p = X.shape
    
    # create data matrix by block
    one_vec = np.ones((n,1))
    big_X = np.concatenate([one_vec, X], axis = 1)
    
    # compute estimator
    beta = ordinary_least_squares(big_X, Y, print_option)
    
    # return result
    return beta
    

def ridge_regression(X, Y, mu, print_option = True):
    """apply ridge regression on (X,Y) with regularization parameter mu
    
    :param X predictors data
    :param Y result
    :param mu L2-regularization parameter
    :param print_option 
    :return: estimator"""
    
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
    
    # concatenate in big vector
    zero_vec = np.zeros(p)
    big_Y = np.concatenate([Y_vec, zero_vec])
    
    # apply ordinary least squares on big matrices
    estimator = ordinary_least_squares(big_X, big_Y)
    
    # retru result
    return estimator

def exterior_derivative_estimation(X, Y, mu, d, print_option = True):
    """apply exterior derivatice regression on (X,Y) with regularization parameter mu and dimension parameter d
    
    :param X predictors data
    :param Y result
    :param mu L2-regularization parameter
    :param d number of dimensions kept from SVD
    :param print_option 
    :return: estimator"""
    
    # perform singular value decomposition
    U, S, V = slg.svd(X)
    
    
    
    

X, Y = getData(filename)    
ordinary_least_squares_with_constant(X, Y)
ridge_regression(X, Y, 1.5)

