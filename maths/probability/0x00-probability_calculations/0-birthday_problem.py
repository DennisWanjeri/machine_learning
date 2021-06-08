#!/usr/bin/python3
"""Calculate the chances of two people having the same age for a group of people"""


#define max group size
n = 30
# number of days in a year
days = 365
#calculate probability for different group sizes
p = 1.0
for i in range(1, n):
    av = days - i
    p *= av/days
    print('n=%d, %d/%d, p=%.3f 1-p=%.3f' % (i+1, av, days, p*100, (1-p)*100))
