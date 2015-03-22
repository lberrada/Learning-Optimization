# coding utf-8

import pandas.io as pio
import numpy.linalg as nlg
import scipy.linalg as slg
import statsmodels.api as sap
import pandas as pd
import numpy as np
import sklearn.linear_model as slm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    X = df[predictors]
    Y = df["quality"]
    
#     # add constant to X
#     X = sap.add_constant(X)
    return X, Y

def get_error(X, Y, beta, shape):
    """compute adjusted error given data, coefficients and data size"""
        
#     # build coefficient array
#     beta = np.array([])
#     for key in est.keys():
#         beta += [est[key]]    
        
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
    return beta, r2

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

def cross_validate(method, X, Y, training_fraction=0.8):
    """perform cross validation for specified method"""
    
    print "Cross validation for " + method + " method"
    
    best_r2 = 0
    best_mu = 0
    best_d = None
    best_est = dict()
    
    n, p = X.shape
    n_train = int(0.8 * n)
    n_test = n - n_train
    shape = (n_test, p)
    
    X_train = X[:n_train]
    Y_train = Y[:n_train]
    X_test = X[n_train:]
    Y_test = Y[n_train:]
    
#     one_train = np.ones((n_train,1))
    one_test = np.ones((n_test,1))
    X_train = pd.DataFrame(X_train, columns = X.keys().tolist())
    X_test = pd.DataFrame(np.concatenate([one_test, X_test], axis = 1), columns = ["const"]+X.keys().tolist())
    
    mu_values = np.arange(0,10,1)
    
    if method == "EDE":
        d_values = np.arange(1,X.shape[1])
        r2_values = np.zeros(len(mu_values) * len(d_values))
    else:
        r2_values = np.zeros(len(mu_values))
    
    ind=0
    
    if method == "lasso":
        for mu in mu_values:
            beta, _ = lasso_regression(X_train, Y_train, mu, print_option = False)
            r2_values[ind] = get_error(X_test, Y_test, beta, shape)
            if r2_values[ind] > best_r2:
                best_r2 = r2_values[-1]
                best_mu = mu
                best_est = beta
            ind += 1
        
    elif method == "least squares":
        print "No parameter to tune, no need for cross-validation"
        
    elif method == "ridge":
        for mu in mu_values:
            beta, _ = ridge_regression(X_train, Y_train, mu, print_option = False)
            r2_values[ind] = get_error(X_test, Y_test, beta, shape)
            if r2_values[ind] > best_r2:
                best_r2 = r2_values[ind]
                best_mu = mu
                best_est = beta
            ind += 1
        
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
    
    print "Best error: ", best_r2
    print "Best mu: ", best_mu
    if best_d != None:
        print "Best d: ", best_d
    
    if method != "EDE":
        
        # plot Adjusted R2 against mu
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(mu_values, r2_values)
        ax.set_xlabel('Regularization Parameter')
        ax.set_ylabel('Adjusted $R^2$')
        plt.title("Cross Validation for " + method + " regression")
        plt.show()
        
    else:
        # plot Adjusted R2 against mu and d
        fig = plt.figure(figsize=(10, 7))
        ax = fig.gca(projection='3d')
        x = mu_values
        y = d_values
        X, Y = np.meshgrid(x, y)
        Z = r2_values
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                linewidth=0, antialiased=False)
        
#         ax.set_zlim(0.51, 0.53)
        ax.set_xlabel('Regularization Parameter')
        ax.set_ylabel('Dimension Parameter')
        ax.set_zlabel('Adjusted $R^2$')
        ax.zaxis.set_major_locator(plt.LinearLocator(10))
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
        
        fig.colorbar(surf, shrink=0.5, aspect=7, cmap=plt.cm.coolwarm)
        
        plt.show()
        
    return best_est
    

X, Y = getData(filename)
cross_validate('ridge', X, Y)