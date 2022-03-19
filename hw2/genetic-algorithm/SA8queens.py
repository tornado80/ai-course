import numpy as np
import random
import math


# INITIAL DATA & PARAMETERS
n = 8
T = 10000
t_change = 0.999


# OBJECTIVE FUNCTION
def objective(sol):
    q = 0
    for i in range(n):
        k = n - 1
        for j in range(n):
            if i != j:
                if sol[i] == sol[j] or abs(i-j) == abs(sol[i]-sol[j]):
                    k = k-1
        q = q + k
    return q/2


# INITIAL SOLUTION
solution = random.sample(range(0, n), n)
fitness = objective(solution)


# MAIN LOOP OF SA ALGORITHM
while T > 0:
    neighbour = solution.copy()
    temp = np.random.randint(n)
    neighbour[temp] = np.random.randint(n)
    fit = objective(neighbour)

    delta = fit - fitness
    if delta >= 0:
        solution = neighbour
        fitness = fit
    else:
        pr = math.exp(delta / T)
        if pr >= .999:
            solution = neighbour
            fitness = fit

    print(fitness)
    T = int(T*t_change)


print(solution)
print(fitness)
