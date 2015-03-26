# coding utf-8

import numpy as np
import numpy.linalg as nlg

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
    adj_r2 = r2 - (1 - r2) * p / (n - p - 1)
        
    # return result
    return adj_r2