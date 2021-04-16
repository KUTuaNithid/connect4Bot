import numpy as np


class ZeroBrain:
    def __init__(self, game):
        self.game = game

    def predict(self, s):
        # P, V = self.model.predict(s)
        P = np.random.dirichlet(np.ones(7))
        V = np.random.uniform(-1,1)
        return P, V

    def train(self, memory):
        return

