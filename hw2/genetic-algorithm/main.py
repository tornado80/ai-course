import math
import random
import time
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(self, file_name: str, population_size: int, mutation_coefficient: float, elapsed_time: float):
        self.read_graph(file_name)
        self.population_size = population_size
        self.elapsed_time = elapsed_time
        self.mutation_coefficient = mutation_coefficient

    def fitness(self, chromosome: list):
        total = 0
        for i in range(len(chromosome)):
            for j in range(i + 1, len(chromosome)):
                if chromosome[j] in self.graph[chromosome[i]]:
                    total += 1
        return total

    def reproduce(self, chromosome_x: list, chromosome_y: list):
        cut = random.randint(1, self.vertex_count - 1)
        child_chromosome = chromosome_x[:cut]
        temp = set(child_chromosome)
        for gene in chromosome_y:
            if gene not in temp:
                child_chromosome.append(gene)
        return child_chromosome, self.fitness(child_chromosome)

    def create_population(self):
        population = []
        for i in range(self.population_size):
            chromosome = random.sample(range(1, self.vertex_count + 1), k=self.vertex_count)
            population.append((chromosome, self.fitness(chromosome)))
        return population

    def read_graph(self, file_name):
        with open(file_name) as f:
            lines = f.readlines()
            self.vertex_count = int(lines[0])
            self.edge_count = 0
            self.graph = {}
            for vertex in range(1, self.vertex_count + 1):
                self.graph[vertex] = set()
            for line in lines[1:]:
                x, y = map(int, line.split())
                self.edge_count += 1
                self.graph[x].add(y)

    def mutate(self, chromosome: list, chromosome_fitness: int):
        i, j = sorted(random.sample(range(len(chromosome)), k=2))
        delta = 0
        if chromosome[i] in self.graph[chromosome[j]]:
            delta = 1
        if chromosome[j] in self.graph[chromosome[i]]:
            delta = -1
        for k in range(i + 1, j):
            if chromosome[i] in self.graph[chromosome[k]]:
                delta += 1
            if chromosome[j] in self.graph[chromosome[k]]:
                delta -= 1
            if chromosome[k] in self.graph[chromosome[i]]:
                delta -= 1
            if chromosome[k] in self.graph[chromosome[j]]:
                delta += 1
        successor = chromosome.copy()
        successor[i], successor[j] = successor[j], successor[i]
        return successor, chromosome_fitness + delta

    def simulate(self):
        def mutate_or_not():
            return random.random() < self.mutation_coefficient
        population = self.create_population()
        begin = time.time()
        plot_data = []
        generation = 0
        global_best, global_best_fitness = population[0], population[0][1]
        while time.time() - begin < self.elapsed_time:
            generation += 1
            population.sort(key=lambda x: x[1], reverse=True)
            local_best = population[0]
            local_best_fitness = local_best[1]
            plot_data.append(local_best_fitness)
            if local_best_fitness > global_best_fitness:
                global_best, global_best_fitness = local_best, local_best_fitness
            if local_best_fitness == self.edge_count:
                break
            new_population = []
            population_weights = [x[1] for x in population]
            for i in range(self.population_size):
                (chromosome_x, _), (chromosome_y, _) = random.choices(population, weights=population_weights, k=2)
                child_chromosome, child_fitness = self.reproduce(chromosome_x, chromosome_y)
                if mutate_or_not():
                    child_chromosome, child_fitness = self.mutate(child_chromosome, child_fitness)
                new_population.append((child_chromosome, child_fitness))
            population = new_population
        return global_best, plot_data, generation


if __name__ == "__main__":
    ga = GeneticAlgorithm("input.txt", 1000, 0.3, 5)
    solution, plot_data, generation = ga.simulate()
    print(f"Best solution found: {solution[0]} with fitness {solution[1]} after {generation} generations.")
    plt.plot(plot_data)
    plt.show()
