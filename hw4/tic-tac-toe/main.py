from agent import Agent
from human import Human
from player import Player
from tic_tac_toe import TicTacToe


def get_symbol_from_user():
    while True:
        symbol = input("Which symbol do you choose? [x/o/(blank for x)]: ")
        if symbol == "" or symbol == "x":
            return "x"
        elif symbol == "o":
            return "o"
        else:
            print("Invalid symbol!", end=" ")


def get_players_from_user(player1: Player, player2: Player) -> (Player, Player):
    while True:
        first_player = input(f"Who shall begin? ([{player1.name[0]}]{player1.name[1:]}/"
                             f"[{player2.name[0]}]{player2.name[1:]}/blank for {player1.name}): ")
        if first_player == player1.name[0] or first_player == "":
            return player1, player2
        elif first_player == player2.name[0]:
            return player2, player1
        else:
            print("Invalid input!", end=" ")


def get_initial_state():
    print("Program will read up to 9 symbols among 'x', 'o' and '-' (without quotes) where '-' represents blank.\n"
          "Leave at least one whitespace between every two adjacent symbols.\n"
          "You can key in symbols in multiple lines too.")
    state = []
    while len(state) < 9:
        symbols = input(f"Enter {9 - len(state)} symbols: ").split()
        for symbol in symbols:
            if symbol not in ['x', 'o', '-']:
                print("Invalid symbol detected!", end="")
                return get_initial_state()
        state.extend(symbols)
    if len(state) > 9:
        print("More than 9 symbols entered and discarded the rest!")
        state = state[:9]
    return state


def prompt_for_initial_state():
    response = input("Do you want the game starts with an initial state? (blank for no otherwise yes): ")
    if response == "":
        return None
    return get_initial_state()


human_symbol = get_symbol_from_user()
human = Human(human_symbol)
agent = Agent(TicTacToe.get_opponent_symbol(human_symbol))
initial_state = prompt_for_initial_state()
player1, player2 = get_players_from_user(human, agent)
game = TicTacToe(player1, player2, initial_state)
print("Created the game. Let's GO!")
game.run()
if game.winner:
    print(f"{game.winner.name} won!")
else:
    print("Game resulted in draw!")
