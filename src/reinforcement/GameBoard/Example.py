from GameBoard import Connect4Board

board = Connect4Board(first_player=1) # first_player = 1 or first_player = 2 
print("Round No : {}".format(board.round))
print("This is what board does look like")
board.showBoard()
print("List of valid action")
print(board.validAction())
print("state with current player ")
print(board.getStateAsPlayer())
x = int(input("enter column (0 - 6) of playerNo {0} : ".format(board.current_turn)))
board.insertColumn(x)
while(board.isEnd is not True):
    print("Round No : {}".format(board.round))
    print("This is what board does look like")
    board.showBoard()
    print("List of valid action")
    print(board.validAction())
    print("state with current player ")
    print(board.getStateAsPlayer())
    print("state for player 1")
    print(board.getState(1).shape)
    print("state for player 2")
    print(board.getState(2).shape)
    x = int(input("enter column (0 - 6) of playerNo {0} : ".format(board.current_turn)))
    board.insertColumn(x)
print("Winner is {}".format(board.winner))