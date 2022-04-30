from enum import Enum
from player import Player


class Cell(Enum):
    BLANK = '-'
    CROSS = 'x'
    NOUGHT = 'o'


class TicTacToe:
    def __init__(self, player1: Player, player2: Player, initial_state: list[str] = None):
        if initial_state:
            self.__state: list[str] = initial_state
        else:
            self.__state: list[str] = ['-'] * 9
        self.__player1: Player = player1
        self.__player2: Player = player2
        self.__players: list[Player] = [player1, player2]
        self.__winner: Player = None

    @property
    def winner(self) -> Player:
        return self.__winner

    def run(self):
        turn = 0
        while not self.__is_finished():
            self.__draw_state()
            player = self.__players[turn]
            player.move(self.__state)
            turn = 1 - turn
        self.__draw_state()

    def __draw_state(self):
        print("### BOARD ###")
        for i in range(9):
            for j in range(9):
                print(self.__state[3 * (i // 3) + j // 3], end="")
            print()
        print("#############")

    @staticmethod
    def status(state: list[str]) -> str:
        """
        Returns 'x' when 'x' wins, 'o' when 'o' wins, 'xo' when draw and None when not terminal
        """
        for symbol in ['x', 'o']:
            if state[0] == state[1] == state[2] == symbol:  # first row
                return symbol
            if state[3] == state[4] == state[5] == symbol:  # second row
                return symbol
            if state[6] == state[7] == state[8] == symbol:  # third row
                return symbol
            if state[0] == state[3] == state[6] == symbol:  # first column
                return symbol
            if state[1] == state[4] == state[7] == symbol:  # second column
                return symbol
            if state[2] == state[5] == state[8] == symbol:  # third column
                return symbol
            if state[0] == state[4] == state[8] == symbol:  # major diagonal
                return symbol
            if state[2] == state[4] == state[6] == symbol:  # minor diagonal
                return symbol
        if state.count('-') == 0:
            return 'xo'
        return None

    def __is_finished(self) -> bool:
        res = self.status(self.__state)
        if res is None:
            return False
        else:
            if res == 'xo':
                self.__winner = None
            else:
                self.__winner = self.__get_player_by_symbol(res)
            return True

    def __get_player_by_symbol(self, symbol) -> Player:
        for player in self.__players:
            if player.symbol == symbol:
                return player
