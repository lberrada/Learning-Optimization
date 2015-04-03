# coding utf-8

from get_data import getData
import lasso 
import ols
import ridge
import ede 

filename = "winequality-red.csv"
method = "OLS" #"ridge", "lasso", "OLS", "EDE"
n_folds = 10

X, Y = getData(filename)
labels = list(X.columns.values)
fun_dict = {"lasso": lasso.cross_validate, "OLS" : ols.estimate, "ridge" : ridge.cross_validate, "EDE" : ede.cross_validate}
    
fun_dict[method](X, Y, n_folds)
    