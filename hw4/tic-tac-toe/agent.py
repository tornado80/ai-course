import math
from player import Player, PlayerType
from tic_tac_toe import TicTacToe


class Agent(Player):
    def __init__(self, symbol: str, name: str = "Computer", alpha_beta_pruning=True):
        super().__init__(name, symbol)
        self.__minimax_call_count = 0
        self.__alpha_beta_pruning = alpha_beta_pruning

    def move(self, state: list[str]):
        self.__minimax_call_count = 0
        utility, action = self.__minimax(state, PlayerType.MAX)
        state[action] = self.symbol
        print(f"{self.name} has played {action + 1} after investigating {self.__minimax_call_count} states. "
              f"State utility is {utility}.")

    def __minimax(self, state: list[str], player_type: PlayerType, alpha=-math.inf, beta=math.inf) -> (int, int):
        self.__minimax_call_count += 1
        utility = self.__utility(state)
        if utility is not None:
            return utility, None
        best_action = None
        if player_type == PlayerType.MAX:
            value = -math.inf
            for i in range(9):
                if state[i] != '-':
                    continue
                state[i] = self.symbol
                utility, _ = self.__minimax(state, PlayerType.MIN, alpha, beta)
                state[i] = '-'
                if utility > value:
                    best_action = i
                    value = utility
                if self.__alpha_beta_pruning:
                    if value >= beta:
                        return value, None
                    alpha = max(value, alpha)
            return value, best_action
        elif player_type == PlayerType.MIN:
            value = math.inf
            for i in range(9):
                if state[i] != '-':
                    continue
                state[i] = TicTacToe.get_opponent_symbol(self.symbol)
                utility, _ = self.__minimax(state, PlayerType.MAX, alpha, beta)
                state[i] = '-'
                if utility < value:
                    best_action = i
                    value = utility
                if self.__alpha_beta_pruning:
                    if value <= alpha:
                        return value, None
                    beta = min(value, beta)
            return value, best_action

    def __utility(self, state: list[str]):
        value = TicTacToe.status(state)
        if value is None:
            return None
        if value == 'xo':
            return 0
        if value == self.symbol:
            return 1
        return -1
