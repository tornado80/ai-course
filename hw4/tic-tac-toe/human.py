from player import Player


class Human(Player):
    def __init__(self, symbol: str, name: str = "You"):
        super().__init__(name, symbol)

    def move(self, state: list[str]):
        print(f"{self.name} is to move!")
        while True:
            inp = input("Enter an integer between 1 and 9 representing an empty cell: ")
            if not inp.isdigit():
                print("Input is not integer!", end=" ")
                continue
            pos = int(inp)
            if pos > 9 or pos < 1:
                print("Input is not between 1 and 9!", end=" ")
                continue
            if state[pos - 1] == '-':
                state[pos - 1] = self.symbol
                return pos - 1
            else:
                print("Cell is not empty!", end=" ")
