from othello import Othello
from networks import ResNet
from mcts import MCTS, SPG, MCTSParallel
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
game = Othello()


# model = ResNet(game, 4, 64, device=device)
# 
# mcts = MCTS(game, {'num_searches': 100000, 'dirichlet_epsilon': 0.5, 'dirichlet_alpha': 0.03, 'C': 2}, model)
# p = mcts.search(game.get_initial_state())


model = ResNet(game, 9, 128, device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2,
    'num_searches': 600,
    'num_iterations': 8,
    'num_selfPlay_iterations': 500,
    'num_parallel_games': 100,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.5,
    'dirichlet_alpha': 0.03
}

spGames = [SPG(game) for spg in range(args['num_parallel_games'])]

pmcts = MCTSParallel(game, args, model)
states = np.stack([spg.state for spg in spGames])
neutral_states = game.change_perspective(states, 1)

pmcts.search(states, spGames)
print(spGames)
