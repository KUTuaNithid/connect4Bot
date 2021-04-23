import sys,os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GameBoard'))
from GameBoard import Connect4Board
from players.ZeroPlayer import ZeroPlayer
from brains.ZeroBrain import ZeroBrain

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

ZeroAI =ZeroBrain(2)
board = Connect4Board(first_player=1) # first_player = 1 or first_player = 2 

while(board.isEnd is not True):
    print("Round No : {}".format(board.round))
    print("This is what board does look like for player {}".format(board.current_turn))
    board.showBoard()
    if board.current_turn == 1 :
        x = int(input("enter column (0 - 6) of playerNo {0} : ".format(board.current_turn)))
    else:
        policy,v = ZeroAI.predict(board.getStateAsPlayer())
        x = np.argmax(policy)
        print("policy is : {}".format(policy))
    board.insertColumn(x)
print("Winner is {}".format(board.winner))