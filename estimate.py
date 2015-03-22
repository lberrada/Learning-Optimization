# coding utf-8

from lasso import lasso_regression
from ols import ordinary_least_squares
from ridge import ridge_regression
from ede import exterior_derivative_estimation

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