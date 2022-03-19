import math
import random
import time
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(self,
                 file_name: str,
                 population_size: int,
                 mutation_coefficient: float,
                 selection_coefficient: float,
                 max_population: int,
                 max_generation: int,
                 fitness_scale: float):
        self.vertex_count, self.edge_count, self.graph, self.updated_hunt, self.updated_edge_count = self.read_graph(file_name)
        self.population_size = population_size
        self.fitness_scale = fitness_scale
        self.max_population = max_population
        self.max_generation = max_generation
        self.selection_coefficient = selection_coefficient
        self.mutation_coefficient = mutation_coefficient

    def fitness(self, chromosome: list):
        #return self.count_updated_edges_in_topological_order(chromosome)
        return 2 ** (self.fitness_scale * self.count_updated_edges_in_topological_order(chromosome) / self.updated_edge_count)

    def count_updated_edges_in_topological_order(self, chromosome: list):
        n = self.vertex_count
        c = 0
        for i in range(n):
            for j in range(i, n):
                if (self.updated_hunt[chromosome[j]][chromosome[i]]):
                    c += 1
        return self.updated_edge_count - c

    def count_edges_in_topological_order(self, chromosome: list):
        total = 0
        for i in range(len(chromosome)):
            for j in range(i + 1, len(chromosome)):
                if chromosome[j] in self.graph[chromosome[i]]:
                    total += 1
        return total

    def is_goal(self, chromosome: list):
        return self.edge_count == self.count_edges_in_topological_order(chromosome)

    def reproduce(self, parent1: list, parent2: list):
        cut = random.randint(1, self.vertex_count - 1)
        child1 = parent1[:cut]
        child2 = []
        child1_set = set(child1)
        for gene in parent2:
            if gene not in child1_set:
                child1.append(gene)
            else:  # gene not in child2
                child2.append(gene)
        child2.extend(parent1[cut:])
        return (child1, self.fitness(child1)), (child2, self.fitness(child2))

    def reproduce1(self, par1, par2):
        n = self.vertex_count
        lcs = [[0 for _ in range(self.vertex_count)] for _ in range(self.vertex_count)]  # longest common substring
        prev = [[0 for _ in range(self.vertex_count)] for _ in range(self.vertex_count)]  # longest common substring
        for i in range(self.vertex_count):
            for j in range(self.vertex_count):
                val = par2[j]
                ind = par1.index(val)
                if (j > 0):
                    lcs[i][j] = lcs[i][j - 1]
                    prev[i][j] = prev[i][j - 1]
                if (ind <= i):
                    if (ind > 0 and j > 0 and lcs[i][j] < lcs[ind - 1][j - 1] + 1):
                        lcs[i][j] = lcs[ind - 1][j - 1] + 1
                        prev[i][j] = val
                    elif lcs[i][j] == 0:
                        lcs[i][j] = 1
                        prev[i][j] = val
        real_lcs = []
        iter = 0
        i, j = n - 1, n - 1
        while iter < lcs[n - 1][n - 1]:
            real_lcs.insert(0, prev[i][j])
            i, j = par1.index(prev[i][j]) - 1, par2.index(prev[i][j]) - 1
            iter += 1
        real_lcs_set = set(real_lcs)
        child1 = []
        complement = []
        for gene in par2:
            if gene not in real_lcs_set:
                complement.append(gene)
        complement_iter = 0
        for gene in par1:
            if gene in real_lcs_set:
                child1.append(gene)
            else:
                child1.append(complement[complement_iter])
                complement_iter += 1

        child2 = []
        complement = []
        for gene in par1:
            if gene not in real_lcs_set:
                complement.append(gene)
        complement_iter = 0
        for gene in par2:
            if gene in real_lcs_set:
                child2.append(gene)
            else:
                child2.append(complement[complement_iter])
                complement_iter += 1
        return (child1, self.fitness(child1)), (child2, self.fitness(child2))

    def reproduce2(self, par1, par2):
        sum_index = []
        for i in range(1, self.vertex_count + 1):
            sum_index.append((par1.index(i) + par2.index(i), i))
        sum_index.sort()
        child = []
        for i in range(self.vertex_count):
            child.append(sum_index[i][1])
        return child, self.fitness(child)

    def create_population(self):
        population = []
        for i in range(self.population_size):
            chromosome = random.sample(range(1, self.vertex_count + 1), k=self.vertex_count)
            population.append((chromosome, self.fitness(chromosome)))
        return population

    def read_graph(self, file_name):
        with open(file_name) as f:
            lines = f.readlines()
            vertex_count = int(lines[0])
            edge_count = 0
            graph = {}
            for vertex in range(1, vertex_count + 1):
                graph[vertex] = set()
            for line in lines[1:]:
                x, y = map(int, line.split())
                edge_count += 1
                graph[x].add(y)
        n = vertex_count
        updated_hunt = [[False for _ in range(n + 1)] for _ in range(n + 1)]  # Matrix n * n (from 1 to n+1)
        m1 = 0  # The counter for count the "True" elements in "updated_hunt"
        for i in range(1, n + 1):
            hunter = [i]
            while hunter:
                head = hunter[0]
                for j in range(1, n + 1):
                    if head in graph[j] and updated_hunt[j][i] == False:
                        updated_hunt[j][i] = True
                        m1 += 1
                        hunter.append(j)
                hunter.pop(0)
        return vertex_count, edge_count, graph, updated_hunt, m1

    def mutate(self, chromosome: list):
        i, j = random.sample(range(len(chromosome)), k=2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome, self.fitness(chromosome)

    def simulate(self):
        def mutate_or_not():
            return random.random() < self.mutation_coefficient
        population = self.create_population()
        plot_data = []
        generation = 0
        global_best_chromosome, global_best_fitness = population[0]
        global_best_generation = 0
        while generation < self.max_generation:
            generation += 1
            population.sort(key=lambda x: x[1], reverse=True)
            population = population[:self.max_population]
            selection_count = math.floor(self.selection_coefficient * len(population))
            local_best_chromosome, local_best_fitness = population[0]
            plot_data.append(self.count_edges_in_topological_order(local_best_chromosome))
            if local_best_fitness > global_best_fitness:
                global_best_chromosome, global_best_fitness = local_best_chromosome, local_best_fitness
                global_best_generation = generation
            if self.is_goal(local_best_chromosome):
                break
            new_population = []
            population_weights = [x[1] for x in population]
            for i in range(selection_count):
                (parent1, _), (parent2, _) = random.choices(population, weights=population_weights, k=2)
                child1, child2 = self.reproduce(parent1, parent2)
                new_population.extend([child1, child2])
            for i in range(len(new_population)):
                chromosome, fitness = new_population[i]
                if mutate_or_not():
                    new_population[i] = self.mutate(chromosome)
            population.extend(new_population)
        return global_best_chromosome, global_best_fitness, global_best_generation, plot_data, generation


if __name__ == "__main__":
    ga = GeneticAlgorithm(
        file_name="input.txt",
        population_size=1000,
        max_population=2,
        mutation_coefficient=0.7,
        selection_coefficient=0.8,
        max_generation=1000,
        fitness_scale=20
    )
    while True:
        best_chromosome, best_fitness, best_generation, plot_data, total_generation = ga.simulate()
        print(f"Best solution found: {best_chromosome} "
              f"with fitness {ga.count_edges_in_topological_order(best_chromosome)} "
              f"after {best_generation} generations in total {total_generation} generations.")
        if best_generation <= 3:
            break
    #plt.plot(plot_data)
    #plt.show()
