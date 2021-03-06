#!/usr/bin/python3
"""ID3 Classification"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np

df = pd.DataFrame()
df['Outlook'] = [
    'sunny', 'sunny', 'overcast', 'rain', 'rain', 'rain',
    'overcast', 'sunny', 'sunny', 'rain', 'sunny',
    'overcast', 'overcast', 'rain'
]
df['Temperature'] = [
    'hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool',
    'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild',
]
df['Humidity'] = [
    'high', 'high', 'high', 'high', 'normal', 'normal', 'normal',
    'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'
]
df['Windy'] = [
    'Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak',
    'Weak', 'Weak',
    'Strong', 'Strong', 'Weak', 'Strong'
]
df['Decision'] = [
    'N', 'N', 'P', 'P', 'P', 'N', 'P', 'N', 'P', 'P',
    'P', 'P', 'P', 'N'
]
print(df)

# calculate the entropy of the Decision Column
# probability of P
p_p = len(df.loc[df.Decision == 'P']) / len(df)
# probability of N
p_n = len(df.loc[df.Decision == 'N']) / len(df)

entropy_decision = -p_n * np.log2(p_n) - p_p * np.log2(p_p)
print('H(S) = {}'.format(entropy_decision))


def f_entropy_decision(data):
    """aids in entropy calculation"""
    p_p = len(data.loc[data.Decision == 'P']) / len(data)
    p_n = len(data.loc[data.Decision == 'N']) / len(data)
    return -p_n * np.log2(p_n) - p_p * np.log2(p_p)

IG_decision_Outlook = entropy_decision #H(S)
# overrall equation
overall_eqn = 'Gain(Decision, Outlook) = Entropy(Decision)'

# Iterate through the values for outlook and compute prob. and entropy values
for name, Outlook in df.groupby('Outlook'):
    num_p = len(Outlook.loc[Outlook.Decision == 'P'])
    num_n = len(Outlook.loc[Outlook.Decision != 'P'])
    num_Outlook = len(Outlook)
    print('p(Decision=P|Outlook={}) = {}/{}'.format(name, num_p, num_Outlook))
    print('p(Decision=N|Outlook={}) = {}/{}'.format(name, num_n, num_Outlook))
    print('p(Decision|Outlook={}) = {}/{}'.format(name, num_Outlook, len(df)))
    print('Entropy(Decision|Outlook={}) = '.format(name) + 
          '_{}/{}.log2({}/{}) - '.format(num_p, num_Outlook, num_p, num_Outlook) +
          '{}/{}.log2({}/{})'.format(num_n, num_Outlook, num_n, num_Outlook))
    entropy_decision_outlook = 0
    # Cannot compute log of 0 hence add checks
    if num_p != 0:
        entropy_decision_outlook -= (num_n / num_Outlook)\
                                    * np.log2(num_p / num_Outlook)

    if num_n != 0:
        entropy_decision_outlook -= (num_n / num_Outlook)\
                                    * np.log2(num_n / num_Outlook)

    IG_decision_Outlook -= (num_Outlook / len(df)) * entropy_decision_outlook
    print()
    overall_eqn += ' - p(Decision|Outlook={}).'.format(name)
    overall_eqn += 'Entropy(Decision|Outlook={})'.format(name)
print(overall_eqn)
print('Gain(Decision, Outlook) = {}'.format(IG_decision_Outlook))

def IG(data, column, ent_decision=entropy_decision):
    """equation of information gain wrapped in a function"""
    IG_decision = ent_decision
    for name, temp in data.groupby(column):
        p_p = len(temp.loc[temp.Decision == 'P']) / len(temp)
        p_n = len(temp.loc[temp.Decision != 'P']) / len(temp)

        entropy_decision = 0

        if p_p != 0:
            entropy_decision -= (p_p) * np.log2(p_p)

        if p_n != 0:
            entropy_decision -= (p_n) * np.log2(p_n)

        IG_decision -= (len(temp) / len(df)) * entropy_decision
    return IG_decision

print('-' * 15)
for col in df.columns[:-1]:
    print("Gain(Decision, {}) = {}".format(col, IG(df, col)))

# Outlook is the maximum hence the root splitting point
for name, temp in df.groupby('Outlook'):
    print('-' * 15)
    print(name)
    print('-' * 15)
    print(temp)
    print('-' * 15)

# remove overcast
df_next = df.loc[df.Outlook != 'overcast']
print(df_next)

# sunny samples
df_sunny = df_next.loc[df_next.Outlook == 'sunny']
# recompute entropy for the sunny samples
entropy_decision = f_entropy_decision(df_sunny)
print(entropy_decision)
for col in df_sunny.columns[1:-1]:
    print('Gain(Decision, {}) = {}'.format(col, IG(df_sunny, col, entropy_decision)))

# we split using Humidity
for name, temp in df_sunny.groupby('Humidity'):
    print('-' * 15)
    print(name)
    print('-' * 15)
    print(temp)
    print('-' * 15)

# we then split the rainy branch
df_rain = df_next.loc[df_next.Outlook == 'rain']
entropy_decision = f_entropy_decision(df_rain)
print(entropy_decision)
# Repeating the gain calculation using rain subset
for col in df_rain.columns[1:-1]:
    print('Gain(Decision, {}) = {}'.format(col, IG(df_rain, col, entropy_decision)))

# splitting using Windy values
for name, temp in df_rain.groupby('Windy'):
    print('-' * 15)
    print(name)
    print('-' * 15)
    print(temp)
    print('-' * 15)
