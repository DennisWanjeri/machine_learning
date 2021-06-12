#!/usr/bin/python3
"""using cdf for the binomial.distribution"""
from scipy.stats import binom


p = 0.3
k = 100

dist = binom(k, p)
#calculate the probability of <=n successes
for n in range(10, 110, 10):
    print("p of {} success: {:.3f}".format(n, dist.cdf(n) * 100))
