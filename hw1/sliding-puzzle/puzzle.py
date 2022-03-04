from enum import Enum
from copy import deepcopy
from prettytable import PrettyTable
from typing import List
from adaptable_heap_priority_queue import AdaptableHeapPriorityQueue


class Action(Enum):
    Up = 1
    Down = 2
    Right = 3
    Left = 4


class State:
    def __init__(self, state_matrix: List[List]):
        union = []
        for row in state_matrix:
            union.extend(row)
        self.__state_tuple: tuple = tuple(union)
        self.__state_matrix = state_matrix

    def to_tuple(self) -> tuple:
        return self.__state_tuple

    def __hash__(self):
        return hash(self.__state_tuple)

    def __eq__(self, other):
        return other.to_tuple() == self.__state_tuple

    def find_position(self, x) -> (int, int):
        for i in range(len(self.__state_matrix)):
            if x in self.__state_matrix[i]:
                return i, self.__state_matrix[i].index(x)

    def __actions(self) -> List[Action]:
        i, j = self.find_position(None)  # find the position of blank cell which is None
        n = len(self.__state_matrix) - 1
        result = []
        if j != 0:
            result.append(Action.Left)
        if j != n:
            result.append(Action.Right)
        if i != 0:
            result.append(Action.Up)
        if i != n:
            result.append(Action.Down)
        return result

    def adjacent_states(self):
        i, j = self.find_position(None)  # find the position of blank cell which is None
        result = []
        for action in self.__actions():
            x = deepcopy(self.__state_matrix)
            if action == Action.Up:
                x[i - 1][j], x[i][j] = x[i][j], x[i - 1][j]
            elif action == Action.Down:
                x[i + 1][j], x[i][j] = x[i][j], x[i + 1][j]
            elif action == Action.Right:
                x[i][j + 1], x[i][j] = x[i][j], x[i][j + 1]
            elif action == Action.Left:
                x[i][j - 1], x[i][j] = x[i][j], x[i][j - 1]
            result.append((State(x), action))
        return result

    def pretty_print(self):
        t = PrettyTable()
        t.add_rows(self.__state_matrix)
        print(t.get_string(header=False, border=True))


class Node:
    def __init__(self, state: State, g_cost: int, h_cost: int, parent, action: Action):
        self.parent = parent
        self.state: State = state
        self.f_cost: int = g_cost + h_cost
        self.g_cost: int = g_cost
        self.h_cost: int = h_cost
        self.action: Action = action

    def pretty_print(self):
        print(f"Action: {self.action}, F-cost: {self.f_cost}, G-cost: {self.g_cost}, H-cost: {self.h_cost}")
        self.state.pretty_print()


class Puzzle:
    def __init__(self, initial_state_matrix: List[List], goal_state_matrix: List[List]):
        self.__initial_state = State(initial_state_matrix)
        self.__goal_state = State(goal_state_matrix)
        self.__goal_state_cells_positions = {
            cell: self.__goal_state.find_position(cell)
            for cell in self.__goal_state.to_tuple() if cell is not None
        }

    def __heuristic(self, state: State) -> int:
        s = 0
        for cell, goal_position in self.__goal_state_cells_positions.items():
            s += self.__manhattan_distance(state.find_position(cell), goal_position)
        return s

    @staticmethod
    def __manhattan_distance(t1: (int, int), t2: (int, int)) -> int:
        return abs(t1[0] - t2[0]) + abs(t1[1] - t2[1])

    def __is_goal_state(self, state: State) -> bool:
        return self.__goal_state == state

    def __children(self, parent_node: Node) -> List[Node]:
        return [
            self.__child_node(parent_node, child_state, action)
            for child_state, action in parent_node.state.adjacent_states()
        ]

    def __child_node(self, parent_node: Node, child_state: State, action: Action) -> Node:
        return Node(
            child_state,
            parent_node.g_cost + 1,  # step cost is 1
            self.__heuristic(child_state),
            parent_node,
            action
        )

    def solve(self) -> List[Node] | None:
        frontier = AdaptableHeapPriorityQueue()
        explored = set()
        frontier_nodes_by_states = {}
        frontier_locators_by_nodes = {}

        def add_to_frontier_finders(locator, node: Node):
            frontier_nodes_by_states[node.state] = node
            frontier_locators_by_nodes[node] = locator

        def add_to_frontier(key: int, node: Node):
            locator = frontier.add(key, node)
            add_to_frontier_finders(locator, node)

        def remove_from_frontier_finders(node: Node):
            del frontier_nodes_by_states[node.state]
            del frontier_locators_by_nodes[node]

        def update_frontier(locator, key: int, node: Node):
            frontier.update(locator, key, node)
            add_to_frontier_finders(locator, node)

        initial_node = Node(
            self.__initial_state,
            0,
            self.__heuristic(self.__initial_state),
            None,
            None
        )

        add_to_frontier(initial_node.f_cost, initial_node)

        while not frontier.is_empty():
            key, node = frontier.remove_min()
            remove_from_frontier_finders(node)
            if self.__is_goal_state(node.state):
                return self.__solution(node)
            explored.add(node.state)
            for child in self.__children(node):
                if child.state not in explored and child.state not in frontier_nodes_by_states:
                    add_to_frontier(child.f_cost, child)
                elif child.state in frontier_nodes_by_states:
                    suspect = frontier_nodes_by_states[child.state]
                    if suspect.f_cost > child.f_cost:
                        suspect_locator = frontier_locators_by_nodes[suspect]
                        remove_from_frontier_finders(suspect)
                        update_frontier(suspect_locator, child.f_cost, child)
        return None

    def __recursively_find_solution(self, node: Node) -> List[Node]:
        if node.parent is None:
            return [node]
        path = self.__solution(node.parent)
        path.append(node)
        return path

    def __solution(self, goal_node: Node) -> List[Node]:
        return self.__recursively_find_solution(goal_node)


if __name__ == "__main__":
    """
    solution = Puzzle(
        8,
        [
            [7, 2, 4],
            [5, None, 6],
            [8, 3, 1]
        ],
        [
            [None, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ]
    ).solve()
    """
    """
    solution = Puzzle(
        15,
        [
            [3, 9, 4, 8],
            [1, 5, 11, 6],
            [7, 15, None, 14],
            [13, 2, 10, 12]
        ],
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, None]
        ]
    ).solve()
    """
    """
    solution = Puzzle(
        15,
        [
            [13, 6, 10, 9],
            [15, 7, 3, 5],
            [11, None, 8, 4],
            [1, 14, 2, 12]
        ],
        [
            [2, 5, 1, 11],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, None]
        ]
    ).solve()
    """

    solution = Puzzle(
        [
            [5, 6, 7],
            [4, None, 8],
            [3, 2, 1]
        ],
        [
            [1, 2, 3],
            [8, None, 4],
            [7, 6, 5]
        ]
    ).solve()

    for node in solution:
        node.pretty_print()
