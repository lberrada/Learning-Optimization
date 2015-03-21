# coding utf-8

import pandas.io as pio
import numpy.linalg as nlg
import statsmodels.api as sap

filename = "winequality-red.csv"

def getData(filename):
    """get data from csv file"""
    
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
    
    ols = sap.OLS(Y, X)
    estimator = ols.fit()
    if print_option:
        print(estimator.summary())
    
    return estimator

X, Y = getData(filename)    
ordinary_least_squares(X, Y)

