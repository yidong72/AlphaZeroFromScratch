import torch
from othello import Othello
from networks import ResNet
from mcts import MCTS
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import fire


class GameEngine:
    
    def __init__(self, file, args):
        self.args = args
        self.game = Othello()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet(self.game, 9, 128, device)
        self.model.load_state_dict(torch.load(file, map_location=device))
        self.model.eval()
        self.mcts = MCTS(self.game, self.args, self.model)
        self.state = self.game.get_initial_state()
        # 1 for human, -1 for AI
        self.current_turn = 1
    
    def play_human_move(self, x, y):
        if self.state[x, y] != 0:
            return False
        if self.current_turn == -1:
            return False
        for dir in range(8):
            if self.game.check_valid_move(self.state, x, y, dir, self.current_turn):
                action = x * self.game.row_count + y
                self.state = self.game.get_next_state(self.state, action, self.current_turn)
                self.current_turn *= -1
                return True
        return False


    def play_computer_move(self):
        if self.current_turn == 1:
            return False
        valid_moves = self.game.get_valid_moves(self.state, self.current_turn)
        # get neural state
        neutral_state = self.game.change_perspective(self.state, self.current_turn)
        policy = self.mcts.search(neutral_state)
        policy *= valid_moves
        policy /= np.sum(policy)
        action = int(np.argmax(policy))
        self.state = self.game.get_next_state(self.state, action, self.current_turn)
        self.current_turn *= -1

    def is_game_over(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, 27)
        return is_terminal

    
    def reset(self):
        self.state = self.game.get_initial_state()
        self.current_turn = 1
    

    def get_score(self):
        black_score = np.sum(self.state == 1)
        white_score = np.sum(self.state == -1)
        return black_score, white_score
    
