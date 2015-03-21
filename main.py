# coding utf-8

import pandas.io as pio
import numpy.linalg as nlg
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
    
    # add constant to X
    X = sap.add_constant(X)
    return X, Y

def ordinary_least_squares(X,Y, print_option = True):
    """apply ordinary-least-squares regression on (X,Y)
    
    :param X predictors data
    :param Y result
    :param print_option 
    :return estimator (ols type)"""
    
    ols = sap.OLS(Y, X)
    estimator = ols.fit()
    if print_option:
        print(estimator.summary())
    
    return estimator

def ridge_regression(X, Y, mu, print_option = True):
    """apply ridge regression on (X,Y) with regularization parameter mu
    
    :param X predictors data
    :param Y result
    :param mu L2-regularization parameter
    :param print_option 
    :return estimator (ols type)"""
    
    # find n number of rows and p number of cols
    n, p = X.shape
    
    # create numpy arrays for block matrices (X)
    X_matrix = X.as_matrix()
    id = np.identity(n)
    mu_matrix = mu * np.ones((n,p))
    zero_matrix = np.zeros((n,n))
    
    # concatenate in upper and lower block matrices (X)
    upper_block = np.concatenate([id, X_matrix], axis = 1)
    lower_block = np.concatenate([zero_matrix, mu_matrix], axis = 1)
    
    # concatenate in big matrix (X)
    big_X = np.concatenate([upper_block, lower_block], axis =0)
    
    # create numpy arrays for block matrices (Y)
    Y_vec = Y.as_matrix()
    zero_vec = np.zeros(n)
    
    # concatenate in big vector
    big_Y = np.concatenate([Y_vec, zero_vec])

    # apply ordinary least squares on big matrices
    estimator = ordinary_least_squares(big_X, big_Y)
    
    return estimator
    

X, Y = getData(filename)    
# ordinary_least_squares(X, Y)
ridge_regression(X, Y, 1)

