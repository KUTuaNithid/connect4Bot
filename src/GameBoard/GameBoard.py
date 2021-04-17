import numpy as np
import copy
class Connect4Board:
    def __init__(self,first_player):
        self.board = np.zeros((6,7),dtype=np.int8)
        self.round = 0
        self.current_turn = first_player
        self.isEnd = False
        self.winner = 0

    def getBoard(self):
        return copy.deepcopy(self.board)

    def showBoard(self):
        print('-------------------')
        for board_line in self.board:
            print("{0}  {1}  {2}  {3}  {4}  {5}  {6}".format(*board_line))
        print('-------------------')

    def getState(self,playerNo,oneHot = True):
        if oneHot:
            CH1 = np.zeros((6,7),dtype=np.int8) 
            CH1[self.board == 0] = 1 # 1 at 0 (6x7)
            CH2 = np.zeros((6,7),dtype=np.int8)
            CH2[self.board == playerNo] = 1 # 1 at playerNo (6x7)
            CH3 = np.zeros((6,7),dtype=np.int8)
            CH3[(self.board !=0)&(self.board != playerNo)] = 1 # 1 at not playerNo and not 0 (6x7)
            board = np.stack((CH1,CH2,CH3),axis = 0) # 3 x 6 x 7 one hot encoded
        else:
            board = copy.deepcopy(self.board)
            board[ (board != playerNo) & (board != 0) ] = -1
            board[ board == playerNo ] = 1
        return board
    
    def getStateAsPlayer(self):
        CH1 = np.zeros((6,7),dtype=np.int8) 
        CH1[self.board == 1] = 1 # 1 at 1 (6x7)
        CH2 = np.zeros((6,7),dtype=np.int8)
        CH2[self.board == 2] = 1 # 1 at 2 (6x7)
        CH3 = np.zeros((6,7),dtype=np.int8) # 0 at 3 (6x7) (playerNo1 turn)
        if self.current_turn == 2: # if playerNo2 turn # 1 at 3 (6x7)
            CH3 = np.ones((6,7),dtype=np.int8)
        board = np.stack((CH1,CH2,CH3),axis = 0)
        return board

    def topRowInColumn(self,col):
        i = 5
        if col > 6 or col < 0 :
            return -1
        while i >= 0 and self.board[i,col] != 0:
            i = i-1
        return i
    
    def checkEndGameFromInsert(self,row,col):
        # Horizontal,Vertical,DiagonalLeft,DiagonalRight
        # Horizontal Check
        board = self.board
        coin_count = 0
        row_idx = row
        col_idx = col
        while (not col_idx-1 < 0) and board[row_idx,col_idx-1] == self.current_turn:
            col_idx = col_idx - 1
        while (not col_idx > 6) and board[row_idx,col_idx] == self.current_turn:
            col_idx = col_idx + 1
            coin_count = coin_count + 1
        if coin_count >= 4:
            self.isEnd = True
            return self.current_turn
        #Vertical Check
        coin_count = 0
        row_idx = row
        col_idx = col
        while (not row_idx-1 < 0) and board[row_idx-1,col_idx] == self.current_turn:
            row_idx = row_idx - 1
        while (not row_idx > 5) and board[row_idx,col_idx] == self.current_turn:
            row_idx = row_idx + 1
            coin_count = coin_count + 1
        if coin_count >= 4:
            self.isEnd = True
            return self.current_turn
        #DiagonalLeft Check
        coin_count = 0
        row_idx = row
        col_idx = col
        while (not col_idx-1 < 0) and (not row_idx-1<0) and board[row_idx-1,col_idx-1] == self.current_turn:
            row_idx = row_idx - 1
            col_idx = col_idx - 1
        while (not col_idx > 6) and (not row_idx > 5) and board[row_idx,col_idx] == self.current_turn:
            row_idx = row_idx + 1
            col_idx = col_idx + 1
            coin_count = coin_count + 1
        if coin_count >= 4:
            self.isEnd = True
            return self.current_turn
        #DiagonalRight Check
        coin_count = 0
        row_idx = row
        col_idx = col
        while (not col_idx+1 > 6) and (not row_idx-1<0) and board[row_idx-1,col_idx+1] == self.current_turn:
            row_idx = row_idx - 1
            col_idx = col_idx + 1
        while (not col_idx < 0) and (not row_idx > 5) and board[row_idx,col_idx] == self.current_turn:
            row_idx = row_idx + 1
            col_idx = col_idx - 1
            coin_count = coin_count + 1
        if coin_count >= 4:
            self.isEnd = True
            return self.current_turn
        #No one won yet
        return 0
    def validAction(self):
        valid_action = []
        for i in range(7):
            if self.topRowInColumn(i) >= 0 :
                valid_action.append(i)
        return valid_action
    def insertColumn(self,col):
        targetRow = self.topRowInColumn(col)
        if targetRow == -1 or self.isEnd:
            return False
        else:
            self.board[targetRow,col] = self.current_turn
            self.winner = self.checkEndGameFromInsert(targetRow,col)
            self.round = self.round + 1
            if self.current_turn == 1:
                self.current_turn = 2
            else:
                self.current_turn = 1
            return True