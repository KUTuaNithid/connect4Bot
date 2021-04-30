import sys,os
import numpy as np
from GameBoard.GameBoard import Connect4Board
from players.ZeroPlayer import ZeroPlayer
from brains.EmbeddedZeroBrain import EmbeddedZeroBrain

# red Human = 1 yellow AI = 2
def checkFirstTurn(state):
    print(np.all(state == np.zeros((6,7),dtype=np.int8)))
    if np.all(state == np.zeros((6,7),dtype=np.int8)):
        return 2 # Ai start First
    else:
        return 1 # Human start First
state = np.zeros((6,7),dtype=np.float32)
first_turn_player = checkFirstTurn(state)
board = Connect4Board(first_player=first_turn_player)
fake_board = Connect4Board(first_player=first_turn_player)
model_name = 'saiV2_intmodel.tflite'
ZeroAI = ZeroPlayer(EmbeddedZeroBrain(model_name))
ZeroAI2 = ZeroPlayer(EmbeddedZeroBrain(model_name))
# get state from pond's. Assume
while board.isEnd is not True:
    print('board in program')
    board.showBoard()
    print('board in real-life')
    fake_board.showBoard()
    print(board.current_turn)
    if board.current_turn == 2 or (board.current_turn == 0 and first_turn_player == 2):
        print('AI_Turn')
        action, policy = ZeroAI.act(board)
        fake_board.insertColumn(action)# change to show chosen LED
        input('press button')#waiting for button
    elif board.current_turn == 1 or (board.current_turn == 0 and first_turn_player == 1):
        print('Player_Turn')
        action = int(input("enter column (0 - 6) of playerNo {0} : ".format(board.current_turn)))
        #action, policy = ZeroAI2.act(board)
        fake_board.insertColumn(action)# change to real action
        input('press button')#waiting for button
    else:
        print('something wrong')
        
    board.updateState(fake_board.getBoard())
print("Player {}".format('WIN' if board.winner == 1 else 'LOSE' if board.winner == 2 else 'DRAW'))


