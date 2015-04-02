# coding utf-8

from get_data import getData
from lasso import lasso_regression
from ols import ordinary_least_squares
from ridge import ridge_regression
from ede import exterior_derivative_estimation

filename = "winequality-red.csv"
method = "EDE" #"ridge", "lasso", "OLS", "EDE"
n_folds = 10

X, Y = getData(filename)
labels = list(X.columns.values)
    
if method == "lasso":
    lasso_regression(X, Y, n_folds)
        
elif method == "OLS":
    ordinary_least_squares(X, Y, n_folds)
    
elif method == "ridge":
    ridge_regression(X, Y, n_folds)
    
elif method == "EDE":
    exterior_derivative_estimation(X, Y, n_folds)

    