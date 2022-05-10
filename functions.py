import numpy as np
from scipy.linalg import expm

def build_q(m, n, c, lam, mu, k, v, gamma, nu):
    states_qty = c+1
    matrix = np.zeros(shape=(states_qty,states_qty))
    for i in range(0,states_qty):
        for j in range(0, states_qty):
            if (j<m+2):
                if (i == j - 1):
                    matrix[i, j] = lam + k * v
                if (i == j+1):
                    matrix[i,j] = i * mu

            elif(j<n):
                if (i == j - 1):
                    matrix[i, j] = lam
                if (i == j+1):
                    matrix[i,j] = i * mu
            elif (j < n+1):
                if (i == j - 1):
                    matrix[i, j] = lam
                if (i == j + 1):
                    matrix[i, j] = n * mu + (i-n)*nu
            else:
                if (i == j - 1):
                    matrix[i, j] = lam*gamma
                if (i == j+1):
                    matrix[i,j] = n * mu + (i-n)*nu
    for i in range(0,states_qty):
        summation = np.sum(matrix[i])
        for j in range(0,states_qty):
            if (i==j):
                matrix[i,j] = summation * -1
    return(matrix)

def get_p_vector(matrix):
    #build q nod
    states_qty = len(matrix)
    vector_ones = np.ones(states_qty)
    matrix[:, states_qty - 1] = vector_ones

    e_vector = np.zeros(states_qty)
    e_vector[states_qty-1] =1
    p = np.matmul(e_vector, np.linalg.inv(matrix))
    return(p)

def define_profit_vector(n,c):
    states_qty = c+1
    profit_vector = np.zeros(states_qty)
    for i in range(0,len(profit_vector)):
        if i<n:
            profit_vector[i] = i * 16 - 10*n
        else:
            profit_vector[i] = n*16 - 10*n
    return(profit_vector)

#asumiendo que el profit de 16 es por cada agent trabajando y que el costo de 10 por agente es fijo en todos los estados
def get_longrun_avg_cost(p_vector, cost_vector):
    avg_cost = 0
    for i in range(0,len(p_vector)):
        avg_cost = avg_cost + p_vector[i]*cost_vector[i]
    return(avg_cost)

def get_discounted_cost(alpha, matrix, cost_vector):
    states_qty = len(cost_vector)
    identity = np.identity(states_qty)
    discounted_costs = alpha*identity - matrix
    discounted_costs = np.linalg.inv(discounted_costs)
    discounted_costs = np.matmul(discounted_costs,cost_vector)
    return(discounted_costs)

def calculate_prob(matrix, time):
    qt_matrix = matrix * time
    final_qt_matrix = expm(qt_matrix)
    return(final_qt_matrix)



matrix = build_q(m, n, c, lam, mu, k, v, gamma, nu)