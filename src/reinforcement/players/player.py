from abc import ABC, abstractmethod
import numpy as np

class Player(ABC):
    def __init__(self, name, game):
        self.name = name
    
    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def observe(self, game):
        pass
    
    @abstractmethod
    def train(self):
        pass