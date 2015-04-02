# coding utf-8

from get_data import getData
from cross_validation import cross_validate
from estimate import estimate


filename = "winequality-red.csv"
# option = "estimation"
option = "cross-validation"
method = "ridge" #"ridge", "lasso", "least squares", "EDE"

X, Y = getData(filename)
labels = list(X.columns.values)
# print labels

if option == "estimate":
    print_option = True
    mu = 1
    d = 5
    estimate(method, X, Y, mu, d, print_option)

elif option == "cross-validation":
    training_fraction = 0.8
    best = cross_validate(method, X, Y, training_fraction)
    my_dict = dict()
    for k in range(len(labels)):
        my_dict[labels[k]] = best[k]
    
    print my_dict
    