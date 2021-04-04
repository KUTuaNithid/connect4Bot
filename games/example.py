#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export PYTHONPATH=/home/connect4Bot:$PYTHONPATH
"""

from games.c4Game import C4Game
import numpy as np

game = C4Game(6, 7, isConv=False)

game.newGame()

illActions = game.getIllMoves()
print(game.getStateActionCnt())
actionCnt = game.getStateActionCnt()

lastS = None
lastA = None
while not game.isOver():
	s = game.getCurrentState()
	print(s.reshape((6,7)))
	illActions = game.getIllMoves()
	while True:
		# a = np.random.choice(actionCnt, 1)[0]
		a = np.random.randint(0,6,1)[0]
		if a not in illActions and a<=6:
			break
	print("action", a)

	game.step(a)
	
	lastS = s
	lastA = a
print("end")
print(game.getCurrentState().reshape((6,7)))