from othello import Othello
from networks import ResNet
from mcts import MCTS
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
game = Othello()
model = ResNet(game, 4, 64, device=device)

mcts = MCTS(game, {'num_searches': 1000, 'dirichlet_epsilon': 0.5, 'dirichlet_alpha': 0.03, 'C': 2}, model)
p = mcts.search(game.get_initial_state())