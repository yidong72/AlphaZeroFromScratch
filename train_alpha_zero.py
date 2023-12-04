from alpha_zero import AlphaZeroParallel
from othello import Othello
from networks import ResNet
import torch
import numpy as np


game = Othello()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device)
model.load_state_dict(torch.load("model_7_<othello.Othello object at 0x7f3e308611e0>.pt", map_location=device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# args = {
#     'C': 2,
#     'num_searches': 1024,
#     'num_iterations': 8,
#     'num_selfPlay_iterations': 512,
#     'num_parallel_games': 256,
#     'num_epochs': 4,
#     'batch_size': 128,
#     'temperature': 1.25,
#     'dirichlet_epsilon': 0.25,
#     'dirichlet_alpha': 0.3
# }
args = {
    'C': 2,
    'num_searches': 1024,
    'num_iterations': 10,
    'num_selfPlay_iterations': 1024,
    'num_parallel_games': 1024,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.00,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}


alphaZero = AlphaZeroParallel(model, optimizer, game, args)
alphaZero.learn()
