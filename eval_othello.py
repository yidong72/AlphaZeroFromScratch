import torch
from othello import Othello
from networks import ResNet
from mcts import MCTS
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import fire


def evaluate(player_1, player_2):    
    game = Othello()

    args = {
        'C': 2,
        'num_searches': 1024,
        'dirichlet_epsilon': 0.0,
        'dirichlet_alpha': 0.01,
        'search': True,
        'temperature': 0,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if player_1.split('_')[1].split('.')[0].startswith('b'):
        model = ResNet(game, 18, 256, device)
    else:
        model = ResNet(game, 9, 128, device)
    model.load_state_dict(torch.load(player_1, map_location=device))
    model.eval()

    if player_2.split('_')[1].split('.')[0].startswith('b'):
        model2 = ResNet(game, 18, 256, device)
    else:
        model2 = ResNet(game, 9, 128, device)
    model2.load_state_dict(torch.load(player_2, map_location=device))
    model2.eval()


    if args['search']:
        mcts = MCTS(game, args, model)
        mcts2 = MCTS(game, args, model2)

    player = 1
    state = game.get_initial_state()

    def get_board(state) -> str:
        # balck #, white o, empty .
        output = ''
        for i in range(game.row_count):
            for j in range(game.column_count):
                if state[i][j] == 1:
                    output += '# '
                elif state[i][j] == -1:
                    output += 'o '
                else:
                    output += '. '
            output += '\n'
        return output


    while True:
        print(get_board(state))

        neutral_state = game.change_perspective(state, player)        

        valid_moves = game.get_valid_moves(state, player)
        # if no more valid moves, skip turn
        if np.sum(valid_moves) == 0:
            print("skipping turn")
            player = game.get_opponent(player)
            continue

        if args['search']:
            if player == 1:
                policy = mcts.search(neutral_state)
            elif player == -1:
                policy = mcts2.search(neutral_state)
        else:
            policy, _ = model.predict(state, augment=self.args['augment']) # Not working with the video's implementation

        policy *= valid_moves
        policy /= np.sum(policy)

        if args['temperature'] == 0:
            action = int(np.argmax(policy))
        elif args['temperature'] == float('inf'):
            action = np.random.choice([r for r in range(game.action_size) if policy[r] > 0])
        else:
            policy = policy ** (1 / args['temperature'])
            policy /= np.sum(policy)
            action = np.random.choice(game.action_size, p=policy)

        if valid_moves[action] == 0:
            print("action not valid")
            continue

        state = game.get_next_state(state, action, player)

        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            print('player', player)
            if value * player == 1:
                print("black won")
            elif value * player == -1:
                print("black loss")
            elif value == 0:
                print("draw")
            print(get_board(state))
            # print number of black and white pieces
            print("black:", np.sum(state == 1))
            print("white:", np.sum(state == -1))
            print("black value", value * player)
            # return value * player, which means 1 if first player won, -1 if second player won
            return value * player
        player = game.get_opponent(player)


if __name__ == '__main__':
    fire.Fire(evaluate)
