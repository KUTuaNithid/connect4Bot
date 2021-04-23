"""
export PYTHONPATH=/home/nithid/connect4Bot/src/reinforcement:$PYTHONPATH

"""

from players.player import Player
import numpy as np
import collections
import copy
import logging
import math

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
logger = logging.getLogger(__file__)

CPUCT = 1
BOARD_COL = 7
SEARCH_LOOP = 400

"""
    Brief: Zero player. Train and Play can be done in this class
"""
class ZeroPlayer():
    def __init__(self, brain):
        self.brain = brain

    def act(self, game):
        """
            Brief: Predict action from current state
            Output: action, policy

            Get current state -> MCTS -> Get tree -> Action from most visited child of root

            Add sample (state, action) to memory
            This action is the best from MCTS
            For value will be inserted after game end, because we don't know we will win yet
        """
        # Get current board state
        current_game = copy.deepcopy(game)

        # MCTS
        root_node = self.MCTS(current_game, SEARCH_LOOP, self.brain)

        # Determine action
        policy = self.get_policy(root_node)
        # print("get_policy", root_node.child_num_visit)
        # print(policy)

        return np.random.choice(np.array([0,1,2,3,4,5,6]), p = policy), policy

    def get_policy(self, root, temp=1):
        """
            Brief: Calculate policy from visit time of first child from root

            Output: Prob of each action from root's state
            policy = [0.1, 0.1, 0.3, 0.1, 0.1, 0.2, 0.1]
        """
        return ((root.child_num_visit)**(1/temp))/sum(root.child_num_visit**(1/temp))

    def MCTS(self, game, num_loop, brain):
        """
            Brief: Create tree containing many state of game

            Select Best UCB leaf -> Fetch prediction -> Expand with P -> Update V to parent -> Repeat
        """ 
        root = MCNode(game, parent = DummyNode())

        for i in range(num_loop):
            leaf = root.select_leaf()
            s = leaf.game.getStateAsPlayer()
            # import time
            # start_time = time.time()
            child_prob, value = brain.predict(s)
            # print("brain.predict --- %s seconds ---" % (time.time() - start_time))
            if leaf.game.isEnd == False: # Expand if game does not finish 
                leaf.expand(child_prob)
            leaf.update_value(value)
        return root

"""
    Brief: Dummy to add value and num_visit for root node
"""
class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_value = collections.defaultdict(float)
        self.child_num_visit = collections.defaultdict(float)

"""
    Brief: Monte Carlo Tree Search node
"""
class MCNode():
    def __init__(self, game, move = None, parent = None):
        self.game = game # Current game state
        self.move = move # Received move to become this state
        self.parent = parent # Previous state
        self.children = {} # Next state

        # Get from NN. Prob of each move [0.5 ... 0.1] len=7, sum is 1
        self.child_prob = np.zeros([BOARD_COL], dtype=np.float32)
        # Get from NN. Value to win 1 for player 1 win. -1 for player 2 win
        self.child_value = np.zeros([BOARD_COL], dtype=np.float32)
        # Visit times of node
        self.child_num_visit = np.zeros([BOARD_COL], dtype=np.float32)
        # Valid action on this state
        self.valid_action = []
        
        self.is_expanded = False

    @property
    def num_visit(self):
        """
            Brief: Get number of visit of this node

            Due to we alway keep num_visit of child, so, for current node, we need to get from its parent
        """
        return self.parent.child_num_visit[self.move]

    @num_visit.setter
    def num_visit(self, value):
        """
            Brief: Update visit time of this node
        """
        self.parent.child_num_visit[self.move] = value
    
    @property
    def value(self):
        """
            Brief: Get value time of this node
        """
        return self.parent.child_value[self.move]
    
    @value.setter
    def value(self, value):
        """
            Brief: Update visit time of this node
        """
        self.parent.child_value[self.move] = value

    def child_Q(self):
        """
            Brief: Calculate Q value of all children
            Output: [Q0, Q1, ..., Q6]

            Q =(V/(1 + NumVisit)
        """
        Q = self.child_value/(1+self.child_num_visit)
        return Q
    
    def child_U(self):
        """
            Brief: Calculate U (2nd term of UCB) value of all children
            Output: [U0, U1, ..., U6]

            U = cpuct * P * root(self.num_visit)/(1+child_num_visit)
        """

        U = CPUCT * self.child_prob * (math.sqrt(self.num_visit)/(1+self.child_num_visit))
        return U

    def bestUCB_move(self):
        """
            Brief: Calculate next move(child) having best UCB
        """
        # bestmove: UCB value of each action (child) [0.24835624 0.24531086 0.03135827 0.25006855 0.24153505 0.23852423 0.03021631]
        bestmove = self.child_Q() + self.child_U()
        # self.valid_action: Possible action correspond to current state [0, 1, 2, 3, 4, 5, 6]
        bestmove = self.valid_action[np.argmax(bestmove[self.valid_action])]
        return bestmove

    def get_child(self, move):
        """
            Brief: Create new child with move
        """
        if move not in self.children: # Check if not be added
            board = copy.deepcopy(self.game)
            # Get new state with move
            board.insertColumn(move)
            self.children[move] = MCNode(board, move = move, parent=self)
        return self.children[move]

    def select_leaf(self):
        """
            Brief: Select the best child from max UCB
        """
        current = self
        while current.is_expanded:
            bestmove = current.bestUCB_move()
            current = current.get_child(bestmove)
        return current
    
    def add_dirichlet_noise(self, valid_action, child_prob):
        valid_child_prob = child_prob[valid_action] # select only legal moves
        valid_child_prob = 0.75*valid_child_prob + 0.25*np.random.dirichlet(np.zeros([len(valid_child_prob)], \
                                                                                          dtype=np.float32)+192)
        child_prob[valid_action] = valid_child_prob
        return child_prob
    
    def expand(self, child_prob):
        """
            Brief: Expand new node from child_prob calculated from nn. Only valid action is used
        """
        valid_action = self.game.validAction()
        self.valid_action = valid_action
        if valid_action == []:
            # Terminal state. No need to expand
            self.is_expanded = False
        else:
            self.is_expanded = True
        valid_child_prob = child_prob
        
        # Insert 0 for illegal move
        valid_child_prob[[i for i in range(len(child_prob)) if i not in valid_action]] = 0.000000000
        if self.parent.parent == None: # add dirichlet noise to child_prob in root node
            valid_child_prob = self.add_dirichlet_noise(valid_action,valid_child_prob)
        self.child_prob = valid_child_prob
    
    def update_value(self, value):
        """
            Brief: Update value to parent node

            current.num_visit++ -> Update value -> Point to parent
        """
        current = self
        while current.parent is not None:
            current.num_visit += 1
            if current.game.current_turn == 1:
                current.value += 1*value # Value for O win
            elif current.game.current_turn == 2:
                current.value += -1*value # Value for O lose(X win)
            current = current.parent



