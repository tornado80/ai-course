import math
import random
import time
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(self,
                 file_name: str,
                 population_size: int,
                 mutation_coefficient: float,
                 elapsed_time: float,
                 fitness_scale: float):
        self.vertex_count, self.edge_count, self.graph = self.read_graph(file_name)
        self.population_size = population_size
        self.fitness_scale = fitness_scale
        self.elapsed_time = elapsed_time
        self.mutation_coefficient = mutation_coefficient

    def fitness(self, chromosome: list):
        return math.exp(self.fitness_scale * self.count_edges_in_topological_order(chromosome) / self.edge_count)

    def count_edges_in_topological_order(self, chromosome: list):
        total = 0
        for i in range(len(chromosome)):
            for j in range(i + 1, len(chromosome)):
                if chromosome[j] in self.graph[chromosome[i]]:
                    total += 1
        return total

    def is_goal(self, chromosome: list):
        return self.edge_count == self.count_edges_in_topological_order(chromosome)

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
        i, j = sorted(random.sample(range(len(chromosome)), k=2))
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome, self.fitness(chromosome)

    def simulate(self):
        def mutate_or_not():
            return random.random() < self.mutation_coefficient
        population = self.create_population()
        begin = time.time()
        plot_data = []
        generation = 0
        global_best = population[0]
        global_best_generation = 0
        while time.time() - begin < self.elapsed_time:
            generation += 1
            population.sort(key=lambda x: x[1], reverse=True)
            local_best = population[0]
            local_best_chromosome = local_best[0]
            local_best_fitness = local_best[1]
            plot_data.append(self.count_edges_in_topological_order(local_best_chromosome))
            if local_best_fitness > global_best[1]:
                global_best = local_best
                global_best_generation = generation
            if self.is_goal(local_best_chromosome):
                break
            new_population = []
            population_weights = [x[1] for x in population]
            for i in range(self.population_size):
                (chromosome_x, _), (chromosome_y, _) = random.choices(population, weights=population_weights, k=2)
                child = child_chromosome, child_fitness = self.reproduce(chromosome_x, chromosome_y)
                if mutate_or_not():
                    child = self.mutate(child_chromosome)
                new_population.append(child)
            population = new_population
        return global_best, global_best_generation, plot_data, generation


if __name__ == "__main__":
    ga = GeneticAlgorithm("input.txt", 1000, 0.4, 2, 30)
    solution, solution_generation, plot_data, total_generation = ga.simulate()
    print(f"Best solution found: {solution[0]} with fitness {ga.count_edges_in_topological_order(solution[0])} "
          f"after {solution_generation} generations in total {total_generation} generations.")
    plt.plot(plot_data)
    plt.show()
