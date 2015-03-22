# coding utf-8

from get_data import getData
from cross_validation import cross_validate
from estimate import estimate


filename = "winequality-red.csv"
# option = "estimation"
option = "cross-validation"
method = "lasso"
mu = 1
d = 5
print_option = True
training_fraction = 0.8

X, Y = getData(filename)

if option == "estimate":
    estimate(method, X, Y, mu, d, print_option)
elif option == "cross-validation":
    cross_validate(method, X, Y, training_fraction)