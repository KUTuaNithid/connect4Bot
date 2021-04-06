#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export PYTHONPATH=/home/nithid/connect4Bot/src/reinforcement:$PYTHONPATH
"""

from games.c4Game import C4Game
import numpy as np
from players.humanPlayer import HumanPlayer
from players.testPlayer import TestPlayer

game = C4Game(6, 7, isConv=False)

game.newGame()

illActions = game.getIllMoves()
print(game.getStateActionCnt())
actionCnt = game.getStateActionCnt()
playerHuman = HumanPlayer("Nithid", game)

lastS = None
lastA = None

aiPlayer = TestPlayer('10Hrsc4DDQNNotGood', game)

selector = 0
while not game.isOver():
	s = game.getCurrentState()
	print(s.reshape((6,7)))
	illActions = game.getIllMoves()
    # # Random move
	# while True:
	# 	# a = np.random.choice(actionCnt, 1)[0]
	# 	a = np.random.randint(0,6,1)[0]
	# 	if a not in illActions and a<=6:
    # 		break
	if selector == 0:
		a = playerHuman.act(game)
		selector = 1
	elif selector == 1:
		a = aiPlayer.act(game)
		selector = 0
		
	game.step(a)
	
	lastS = s
	lastA = a
print("end")
print(game.getCurrentState().reshape((6,7)))