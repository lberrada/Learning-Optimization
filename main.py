# coding utf-8

import pandas.io as pio
import numpy.linalg as nlg
import scipy.linalg as slg
import statsmodels.api as sap
import pandas as pd
import numpy as np
import sklearn.linear_model as slm
import matplotlib.pyplot as plt

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

def get_error(X, Y, beta, shape):
    """compute adjusted error given data, coefficients and data size"""
        
    # get data shape
    n, p = shape
    
    # compute sum of square of residuals
    SSR = nlg.norm(Y-np.dot(X,beta)) ** 2
    
    # compute sum of squares
    m = np.mean(Y)
    SS = nlg.norm(Y - m) ** 2
    
    # compute r2
    r2 = 1 - SSR / SS
    
    # compute adjusted r2
    adj_r2 = r2 + (1 - r2) * p / (n - p - 1)
    
    # return result
    return adj_r2
    

def least_squares(X, Y, shape, predictors, print_option = True):
    """apply ordinary-least-squares regression on (X,Y)
    
    :param X predictors data
    :param Y result
    :param print_option 
    :return: est (dictionary)"""

    # compute X'X
    XtX = np.matrix(np.dot(X.transpose(),X))
    
    # compute beta
    beta = np.array(np.dot(np.dot(nlg.inv(XtX),X.transpose()),Y)).flatten()
    
    # compute error
    r2 = get_error(X, Y, beta, shape)
    
    # create estimator dictionary
    est = dict()
    
    # relate to predictors
    for ind in range(len(predictors)):
        est[predictors[ind]] = beta[ind]
        
    # print results if needed
    if print_option:
        print "Error : ", r2
        print est
    
    # return result
    return est, r2

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
    shape = (n,p)
    
    # create data matrix by block
    one_vec = np.ones((n,1))
    big_X = np.concatenate([one_vec, X], axis = 1)
    
    # compute estimator
    est, r2 = least_squares(big_X, Y, shape, predictors, print_option)
    
    # return result
    return est, r2
    

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
    shape = (n, p)
    
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
    est, r2 = least_squares(big_X, big_Y, shape, predictors, print_option)
    
    # retru result
    return est, r2

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
    shape = (n, p)
    
    # create data matrix by block
    one_vec = np.ones((n,1))
    big_X = np.concatenate([one_vec, X], axis = 1)
    
    # compute regression
    clf = slm.Lasso(alpha = mu)
    clf.fit(big_X, Y)
    
    # get estimator
    beta = clf.coef_
        
    # get error
    r2 = get_error(big_X, Y, beta, shape)
    
    # create estimator dictionary
    est = dict()
    
    # relate to predictors
    for ind in range(len(predictors)):
        est[predictors[ind]] = beta[ind]
        
    if print_option:
        print "Error :", r2
        print est
        
    # return result
    return est, r2
    

def exterior_derivative_estimation(X, Y, mu, d, print_option = True):
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
    est, r2 = least_squares(big_X, big_Y, shape, predictors, print_option)
    
    # retru result
    return est, r2

def estimate(method, X, Y, mu = None, d = None, print_option = True):
    """perform linear estimation according to specified method"""
    
    print "Linear estimation with " + method + " method"
    
    if method == "lasso":
        est, r2 = lasso_regression(X, Y, mu, print_option)
        
    elif method == "least squares":
        est, r2 = ordinary_least_squares(X, Y, print_option)
        
    elif method == "ridge":
        est, r2 = ridge_regression(X, Y, mu, print_option)
        
    elif method == "EDE":
        est, r2 = exterior_derivative_estimation(X, Y, mu, d, print_option)
        
    else:
        print "Error: method should be 'lasso', 'least squares', 'ridge' or 'EDE'"
        return
    
    return est, r2

def cross_validate(method, X, Y):
    """perform cross validation for specified method"""
    
    print "Cross validation for " + method + " method"
    
    best_r2 = 0
    best_mu = 0
    best_d = None
    best_est = dict()
    
    mu_values = np.arange(0,10,0.1)
    
    if method == "EDE":
        d_values = np.arange(1,X.shape[1])
        r2_values = np.zeros(len(mu_values) * len(d_values))
    else:
        r2_values = np.zeros(len(mu_values))
    
    ind=0
    
    if method == "lasso":
        for mu in mu_values:
            est, r2_values[ind] = lasso_regression(X, Y, mu, print_option = False)
            if r2_values[ind] > best_r2:
                best_r2 = r2_values[-1]
                best_mu = mu
                best_est = est
            ind += 1
        
    elif method == "least squares":
        print "No parameter to tune, no need for cross-validation"
        
    elif method == "ridge":
        for mu in mu_values:
            est, r2_values[ind] = ridge_regression(X, Y, mu, print_option = False)
            if r2_values[ind] > best_r2:
                best_r2 = r2_values[ind]
                best_mu = mu
                best_est = est
            ind += 1
        
    elif method == "EDE":
        for mu in mu_values:
            for d in d_values:
                est, r2_values[ind] = exterior_derivative_estimation(X, Y, mu, d, print_option = False)
                if r2_values[ind] > best_r2:
                    best_r2 = r2_values[-1]
                    best_mu = mu
                    best_d = d
                    best_est = est
                ind += 1
        
    else:
        print "Error: method should be 'lasso', 'least squares', 'ridge' or 'EDE'"
        return
    
    print "Best error: ", best_r2
    print "Best mu: ", best_mu
    if best_d != None:
        print "Best d: ", best_d
    
    if method != "EDE":
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(mu_values, r2_values)
        ax.set_xlabel('Regularization Parameter')
        ax.set_ylabel('Adjusted $R^2$')
        plt.title("Cross Validation for " + method + " regression")
        plt.show()
        
    return best_est
    

X, Y = getData(filename)
cross_validate('ridge', X, Y)

