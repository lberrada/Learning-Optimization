# coding utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lasso import lasso_regression
from ridge import ridge_regression
from ede import exterior_derivative_estimation
from error import get_error

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