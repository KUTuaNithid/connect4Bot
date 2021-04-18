from players.ZeroPlayer import ZeroPlayer
from brains.ZeroBrain import ZeroBrain
import pickle
from tqdm import tqdm
import datetime
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GameBoard'))
from GameBoard import Connect4Board
from argparse import ArgumentParser

def save_as_pickle(filename, data):
    completeName = os.path.join("./datasets/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def SelfPlay(num_games, iteration):
    if not os.path.isdir("./datasets/iter_%d" % (iteration+1)):
        if not os.path.isdir("datasets"):
            os.mkdir("datasets")
        os.mkdir("datasets/iter_%d" % (iteration+1))
    
    for i in tqdm(range(0, num_games)):
        board = Connect4Board(first_player=1)
        dataset = [] # To train neural network [state, policy, value]
        test_player = ZeroPlayer(ZeroBrain(iteration))
        while(board.isEnd is not True):
            state = board.getStateAsPlayer()
            action, policy = test_player.act(board)
            board.insertColumn(action)
            print("Round No : {}".format(board.round))
            print("This is what board does look like")
            board.showBoard()

            # Get dataset
            dataset.append([state, policy])
        if board.winner == 1:
            value = 1
        elif board.winner == 2:
            value = 2
        else:
            value = 0
        dataset_p = []

        # All move inside make 1 or 2 win
        # So, we can assign the same value
        for idx,data in enumerate(dataset):
            s,p = data
            if idx == 0:
                dataset_p.append([s,p,0])
            else:
                dataset_p.append([s,p,value])
        del dataset
        # Dataset for next iteration training
        save_as_pickle("iter_%d/" % (iteration+1) + "dataset_%d_%s" % (i, datetime.datetime.today().strftime("%Y-%m-%d")), dataset_p)

def train_brain(name):
    pass

def evaluate_brain(net1, net2):
    # Load model net1 and net2
    cur_player = ZeroPlayer(ZeroBrain(net1))
    better_player = ZeroPlayer(ZeroBrain(net2))

    board = Connect4Board(first_player=1)
    while(board.isEnd is not True):
        if board.current_turn == 1:
            action, _ = cur_player.act(board)
        elif board.current_turn == 2:
            action, _ = better_player.act(board)
        board.insertColumn(action)
    if board.winner == 1:
        winner = net1
    elif board.winner == 2:
        winner = net2
    return winner

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_games", type=int, default=100, help="Number of self game play")

    args = parser.parse_args()

    i = 0
    while True:
        # 1. Self play
        print("Self play", i)
        SelfPlay(args.num_games, i)

        # 2. Train next model will be used in next iteration
        print("Training", i+1)
        train_brain(i+1)

        # 3. Evaluate
        if i >= 1: # Already have more than 1 brain
            print("Evaluate", i, i+1)
            winner = evaluate_brain(i, i+1)
            added_game = 10
            while winner != i+1:
                print("Latest model is worse than previous, so retrain with more game", args.num_games+added_game)
                
                # Generate dataset
                SelfPlay(args.num_games+added_game, i)
                # Retrain
                train_brain(i+1)
                winner = evaluate_brain(i, i+1)
                added_game = added_game * 2
        i = i + 1
    