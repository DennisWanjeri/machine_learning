#!/usr/bin/python3
"""calculating moments of a a binomial distribution"""
from scipy.stats import binom



#distribution parameters
p = 0.3
k = 100
#calculate moments
mean, var, _, _ = binom.stats(k, p, moments='mvsk')
print("Mean=%.3f, Variance=%.3f" % (mean, var))
