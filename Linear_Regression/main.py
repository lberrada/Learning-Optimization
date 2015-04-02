# coding utf-8

from get_data import getData
from cross_validation import cross_validate
from estimate import estimate


filename = "winequality-red.csv"
# option = "estimation"
option = "cross-validation"
method = "EDE" #"ridge", "lasso", "least squares", "EDE"

X, Y = getData(filename)
labels = list(X.columns.values)
# print labels

if option == "estimate":
    print_option = True
    mu = 1
    d = 5
    estimate(method, X, Y, mu, d, print_option)

elif option == "cross-validation":
    n_folds = 10
    print "="*50
    print "Cross-Validation"
    print "="*50
    best = cross_validate(method, X, Y, n_folds)
    print "="*50
    print "Coefficients"
    print "="*50
    for key in best.keys():
        print key + " : " + str(best[key])
    