#!/usr/bin/python3
"""probability mass function for our Binomial distribution"""
from scipy.stats import binom


p = 0.3
k = 100
#define distribution
dist = binom(k, p)
#calculate the probability of n successes
for n in range(10, 110, 10):
    print('P of %d success: %.3f%%' % (n, dist.pmf(n)*100))
