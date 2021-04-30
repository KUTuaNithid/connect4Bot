import sys,os
import numpy as np
import time
from reinforcement.GameBoard.GameBoard import Connect4Board
from reinforcement.players.ZeroPlayer import ZeroPlayer
from reinforcement.brains.EmbeddedZeroBrain import EmbeddedZeroBrain
from ImageProcess.Image_processing import ImageProcessing
from GPIO.coral_gpio import GPIO

def checkFirstTurn(state):
    print(np.all(state == np.zeros((6,7),dtype=np.int8)))
    if np.all(state == np.zeros((6,7),dtype=np.int8)):
        return 2 # Ai start First
    else:
        return 1 # Human start First

if __name__ == "__main__":
    #### INITIAL Object ####
    gpio_control = GPIO()
    image_processing = ImageProcessing()

    #### Calibration ####
    print("ready to calibrate")
    gpio_control.wait_push()
    gpio_control.on_all_led()
    image_processing.calibration()
    time.sleep(2)
    gpio_control.off_all_led()

    while(1):
        #### START ####
        gpio_control.wait_push()
        gpio_control.showConfirmButton()
        ## CHECK FIRST PLAYER AND SETUP BOARD
        state = image_processing.process_image()
        print("This is initial state of the board")
        print(state)
        first_turn_player = checkFirstTurn(state)
        board = Connect4Board(first_player=first_turn_player)
        fake_board = Connect4Board(first_player=first_turn_player)

        ## LOAD MODEL ZERO BRAIN
        model_name = 'saiV2_intmodel_edgetpu.tflite'
        ZeroAI = ZeroPlayer(EmbeddedZeroBrain(model_name))
        #ZeroAI2 = ZeroPlayer(EmbeddedZeroBrain(model_name))
        
        #### GAME STARTOOO ####
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
                gpio_control.on_led(action)
                gpio_control.wait_push()
                gpio_control.showConfirmButton()
            elif board.current_turn == 1 or (board.current_turn == 0 and first_turn_player == 1):
                print('Player_Turn')
                #action = int(input("enter column (0 - 6) of playerNo {0} : ".format(board.current_turn)))
                #action, policy = ZeroAI2.act(board)
                if not (board.current_turn == 0 and first_turn_player == 1):
                    fake_board.insertColumn(action)# change to real action
                    gpio_control.wait_push()
                    gpio_control.showConfirmButton()
            else:
                print('something wrong')

            # POND read_state #
            state = image_processing.process_image()
            board.updateState(state)
        print("Player {}".format('WIN' if board.winner == 1 else 'LOSE' if board.winner == 2 else 'DRAW'))
        gpio_control.showWinner(board.winner)