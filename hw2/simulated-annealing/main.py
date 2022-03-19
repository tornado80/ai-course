import math
import random
import matplotlib.pyplot as plt


def read_graph(file_name):
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


def val(graph: dict, order: list):
    total = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if order[j] in graph[order[i]]:
                total += 1
    return total


def random_successor_and_value(graph: dict, current_order: list, current_value: int):
    i, j = sorted(random.sample(range(len(current_order)), k=2))
    delta = 0
    if current_order[i] in graph[current_order[j]]:
        delta = 1
    if current_order[j] in graph[current_order[i]]:
        delta = -1
    for k in range(i + 1, j):
        if current_order[i] in graph[current_order[k]]:
            delta += 1
        if current_order[j] in graph[current_order[k]]:
            delta -= 1
        if current_order[k] in graph[current_order[i]]:
            delta -= 1
        if current_order[k] in graph[current_order[j]]:
            delta += 1
    successor = current_order.copy()
    successor[i], successor[j] = successor[j], successor[i]
    return successor, current_value + delta


def simulated_annealing(vertex_count: int, graph: dict, temperature: int):
    current_order = [i for i in range(1, vertex_count + 1)]
    random.shuffle(current_order)
    current_value = val(graph, current_order)
    # random.random() returns a random number between 0 and 1 with uniform distribution
    decide = lambda probability: random.random() < probability
    print(current_order, current_value)
    plot_data = [current_value]
    while temperature > 1e-6:
        successor, successor_value = random_successor_and_value(graph, current_order, current_value)
        delta = successor_value - current_value
        if delta >= 0 or decide(math.exp(delta / temperature)):
            current_order = successor
            current_value = successor_value
        temperature *= 0.9
        plot_data.append(current_value)
    return current_order, current_value, plot_data


if __name__ == "__main__":
    vertex_count, edge_count, graph = read_graph("input.txt")
    result, result_value, plot_data = simulated_annealing(vertex_count, graph, 100)
    print(result, result_value, edge_count)
    plt.plot(plot_data)
    plt.show()
