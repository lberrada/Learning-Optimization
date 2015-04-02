# coding utf-8

import numpy as np
import numpy.linalg as nlg

def least_squares(X, Y):
    from sklearn.linear_model import LinearRegression
    
    clf = LinearRegression(fit_intercept=False)
    clf.fit(X, Y)
        
    return clf
    