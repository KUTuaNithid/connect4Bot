from GameBoard import Connect4Board

board = Connect4Board(first_player=1)
board.showBoard()
while(board.isEnd is not True):
    x = int(input("enter column (0 - 6) of playerNo {0} : ".format(board.current_turn)))
    board.insertColumn(x)
    print("Round No : {}".format(board.round))
    print("This is what board does look like")
    board.showBoard()
    print("state for player 1")
    print(board.getState(1))
    print("state for player 2")
    print(board.getState(2))
print("Winner is {}".format(board.winner))