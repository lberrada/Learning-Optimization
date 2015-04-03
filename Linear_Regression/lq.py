# coding utf-8

from sklearn.linear_model import LinearRegression

def least_squares(X, Y, predictors):
    """perform least squares regression on (X,Y)"""
    
    clf = LinearRegression(fit_intercept=False)
    clf.fit(X, Y)
    
    beta = clf.coef_

    est = dict()
    ind = 0
    for predictor in predictors:
        est[predictor] = beta[ind]
        ind += 1
        
    return clf, est
    