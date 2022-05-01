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
        self.vertex_count, self.edge_count, self.graph = self.read_graph(file_name)
        self.population_size = population_size
        self.fitness_scale = fitness_scale
        self.max_population = max_population
        self.max_generation = max_generation
        self.selection_coefficient = selection_coefficient
        self.mutation_coefficient = mutation_coefficient

    def fitness(self, chromosome: list):
        return 2 ** (self.fitness_scale * self.count_edges_in_topological_order(chromosome) / self.edge_count)

    def fitness1(self, chromosome: list):
        total = 0
        for i in range(len(chromosome)):
            edge = 0
            for j in range(i + 1, len(chromosome)):
                if chromosome[j] in self.graph[chromosome[i]]:
                    edge += 1
            total += 2 ** (self.vertex_count - i) * edge
        return total

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
        return vertex_count, edge_count, graph

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
            parent_population = population[:selection_count]
            population_weights = [x[1] for x in parent_population]
            for i in range(selection_count):
                (parent1, _), (parent2, _) = random.choices(parent_population, weights=population_weights, k=2)
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
        population_size=100,
        max_population=120,
        mutation_coefficient=0.3,
        selection_coefficient=0.5,
        max_generation=20,
        fitness_scale=20
    )
    best_chromosome, best_fitness, best_generation, plot_data, total_generation = ga.simulate()
    print(f"Best solution found: {best_chromosome} "
          f"with fitness {ga.count_edges_in_topological_order(best_chromosome)} "
          f"after {best_generation} generations in total {total_generation} generations.")
    plt.plot(plot_data)
    plt.show()
