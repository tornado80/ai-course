from abc import ABC, abstractmethod
from enum import Enum


class PlayerType(Enum):
    MAX = 1
    MIN = 2


class Player(ABC):
    def __init__(self, name: str, symbol: str):
        self.__name = name
        self.__symbol = symbol

    @abstractmethod
    def move(self, state: list[str]):
        pass

    @property
    def name(self):
        return self.__name

    @property
    def symbol(self):
        return self.__symbol
