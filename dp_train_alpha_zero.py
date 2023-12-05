from alpha_zero import AlphaZeroParallel
from othello import Othello
from networks import ResNet
import torch
# import numpy as np
import os
from torch import multiprocessing as mp
from torch import distributed as dist


def init_processes(rank, world_size, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def alpha_zero_work(rank, world_size):
    init_processes(rank, world_size)

    game = Othello()
    # get cuda device of rank
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    model = ResNet(game, 9, 128, device)
#    model.load_state_dict(torch.load("model_0_Othello_id140135011983440.pt", map_location=device))

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        'C': 2,
        'num_searches': 1024,
        'num_iterations': 10,
        'num_selfPlay_iterations': 256,
        'num_parallel_games': 256,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.00,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }


    alphaZero = AlphaZeroParallel(model, optimizer, game, args)
    alphaZero.learn()
    
    
if __name__ == "__main__":
    world_size = 2
    mp.spawn(alpha_zero_work, args=(world_size,), nprocs=world_size, join=True)