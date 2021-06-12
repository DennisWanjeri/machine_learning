#!/usr/bin/python3
"""simulating a binomial process and counting success"""
from numpy.random import binomial


#define distribution parameters
p = 0.3
k = 100

#run a single simulation
success = binomial(k, p)
print("Total Success:%d" % success)
