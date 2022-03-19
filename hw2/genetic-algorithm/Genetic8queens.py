import numpy as np

n = 8
population_size = 30
max_pop_size = 50
crossover_coeff = 0.8
mutation_coeff = 0.4
max_iteration = 500
num_crossover = round(population_size*crossover_coeff)
num_mutation = round(population_size*mutation_coeff)
total = population_size+num_crossover+num_mutation

population = []
object_values = []
best_objective = 0
best_chromosome = np.zeros(n)


# objective function
def objective(chrom):
    obj = 0
    for i in range(n):
        k = n-1
        for j in range(n):
            if i != j:
                if chrom[i] == chrom[j] or abs(i-j) == abs(chrom[i]-chrom[j]):
                    k = k-1
        obj = obj + k
    return obj/2


# initial population
while len(population) < population_size:
    temp = np.random.randint(n, size=n).tolist()
    population.append(temp)
    object_values.append(objective(temp))


# main loop of genetic algorithm
iteration = 0
while iteration < max_iteration:
    # roulette wheel
    summation = sum(object_values)
    pr = []
    cumulative_pr = []
    for i in range(population_size):
        pr.append(object_values[i]/summation)
    cumulative_pr.append(pr[0])
    for i in range(1, population_size-1):
        temp = cumulative_pr[i-1] + pr[i]
        cumulative_pr.append(temp)
    cumulative_pr.append(1)

    for i in range(0, int(num_crossover), 2):
        p1 = 0
        temp = np.random.rand()
        while cumulative_pr[p1] < temp:
            p1 = p1 + 1
        p2 = p1
        while p1 == p2:
            temp = np.random.rand()
            p = 0
            while cumulative_pr[p] < temp:
                p = p + 1
            p2 = p

        parent1 = population[p1]
        parent2 = population[p2]
    # crossover
        temp = np.random.randint(n)
        child1 = parent1[0:temp]+parent2[temp:n]
        child2 = parent2[0:temp]+parent1[temp:n]

        population.append(child1)
        object_values.append(objective(child1))
        population.append(child2)
        object_values.append(objective(child2))

    # mutation
    for i in range(num_mutation):
        temp = np.random.randint(num_crossover)
        temp = population_size + temp
        mutated = population[temp]
        temp = np.random.randint(n)
        mutated[temp] = np.random.randint(n)
        population.append(mutated)
        object_values.append(objective(mutated))

    # update best solution
    best_objective = max(object_values)
    best_arg = np.argmax(object_values)
    best_chromosome = population[best_arg]

    # keep best chromosomes
    if len(population) > max_pop_size:
        temp_population = []
        temp_objective = []
        args = np.argsort(object_values)
        for i in range(max_pop_size):
            t = len(population)-1 - i
            temp_population.append(population[args[t]])
            temp_objective.append(object_values[args[t]])
        population = temp_population
        object_values = temp_objective
        population_size = max_pop_size
        
    print(best_objective)
    iteration = iteration + 1

print(best_chromosome)
print(best_objective)






