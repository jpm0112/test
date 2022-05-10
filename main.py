

import numpy as np
import pandas as pd
from functions import *

c= 20
lam = 200
k = 0.5
v= 50
gamma=0.9
mu=15
nu= 5
time=1/6
initial_state = 0
alpha = 0.001

m_values = [5,6,7,8]
n_values = [10,11,12,13,14,15,16,17,18,19,20]

# Part 3

prob_no_wait_table = np.zeros(shape=(len(m_values),len(n_values)))

for m in range(0,len(m_values)):
    for n in range(0,len(n_values)):
        matrix = build_q(m_values[m], n_values[n], c, lam, mu, k, v, gamma, nu)
        probs = calculate_prob(matrix, time)
        prob_no_wait_table[m,n] = np.sum(probs[initial_state][0:n_values[n]])

pd_table = pd.DataFrame(prob_no_wait_table)
pd_table.to_csv('prob_no_wait_table.csv',index=False)


# Part 4


expected_rejected_calls_table = np.zeros(shape=(len(m_values),len(n_values)))
expected_immediate_hangsups_table = np.zeros(shape=(len(m_values),len(n_values)))
expected_not_immediate_hangsups_table = np.zeros(shape=(len(m_values),len(n_values)))

for m in range(0,len(m_values)):
    for n in range(0,len(n_values)):
        matrix = build_q(m_values[m], n_values[n], c, lam, mu, k, v, gamma, nu)
        p_vector = get_p_vector(matrix)

        expected_rejected_calls_table[m,n] = p_vector[c]*lam
        expected_immediate_hangsups_table[m,n] = np.sum(p_vector[n_values[n]:c])*lam*(1-gamma) #sin contar el ultimo estado

        sum = 0
        vector_nu = p_vector[n_values[n]+1:c+1] * nu
        for i in range(0,len(vector_nu)):
            vector_nu[i] = vector_nu[i]*(i+1)

        expected_not_immediate_hangsups_table[m, n] = np.sum(vector_nu) # contando el ultimo estado


pd_table = pd.DataFrame(expected_rejected_calls_table)
pd_table.to_csv('expected_rejected_calls_table.csv',index=False)


pd_table = pd.DataFrame(expected_immediate_hangsups_table)
pd_table.to_csv('expected_immediate_hangsups_table.csv',index=False)


pd_table = pd.DataFrame(expected_not_immediate_hangsups_table)
pd_table.to_csv('expected_not_immediate_hangsups_table.csv',index=False)


# Part 5 and 6


longrun_cost_table = np.zeros(shape=(len(m_values),len(n_values)))
discounted_cost_table = np.zeros(shape=(len(m_values),len(n_values)))


for m in range(0,len(m_values)):
    for n in range(0,len(n_values)):
        matrix = build_q(m_values[m], n_values[n], c, lam, mu, k, v, gamma, nu)
        p_vector = get_p_vector(matrix)
        cost_vector = define_profit_vector(n_values[n], c)
        longrun_cost_table[m,n] = get_longrun_avg_cost(p_vector, cost_vector)



for m in range(0,len(m_values)):
    for n in range(0,len(n_values)):
        matrix = build_q(m_values[m], n_values[n], c, lam, mu, k, v, gamma, nu)
        cost_vector = define_profit_vector(n_values[n], c)
        discounted_cost_table[m, n] = get_discounted_cost(alpha, matrix, cost_vector)[initial_state]




pd_table = pd.DataFrame(longrun_cost_table)
pd_table.to_csv('longrun_cost_table.csv',index=False)


pd_table = pd.DataFrame(discounted_cost_table)
pd_table.to_csv('discounted_cost_table.csv',index=False)







