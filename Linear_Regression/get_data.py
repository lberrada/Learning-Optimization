# coding utf-8

import pandas.io as pio

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