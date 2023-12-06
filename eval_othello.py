import torch
from othello import Othello
from networks import ResNet
from mcts import MCTS
import numpy as np

class Agent:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        if self.args['search']:
            self.mcts = MCTS(self.game, self.args, self.model)

    def run(self, obs, conf):
        player = obs['mark'] if obs['mark'] == 1 else -1
        state = np.array(obs['board']).reshape(self.game.row_count, self.game.column_count)
        state[state==2] = -1
        
        state = self.game.change_perspective(state, player)        

        if self.args['search']:
            policy = self.mcts.search(state)

        else:
            policy, _ = self.model.predict(state, augment=self.args['augment']) # Not working with the video's implementation

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)

        if self.args['temperature'] == 0:
            action = int(np.argmax(policy))
        elif self.args['temperature'] == float('inf'):
            action = np.random.choice([r for r in range(self.game.action_size) if policy[r] > 0])
        else:
            policy = policy ** (1 / self.args['temperature'])
            policy /= np.sum(policy)
            action = np.random.choice(self.game.action_size, p=policy)

        return action
    
game = Othello()

args = {
    'C': 2,
    'num_searches': 800,
    'dirichlet_epsilon': 0.1,
    'dirichlet_alpha': 0.01,
    'search': True,
    'temperature': 0,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device)
model.load_state_dict(torch.load("model_5_Othello_id139744445463376.pt", map_location=device))
model.eval()

if args['search']:
    mcts = MCTS(game, args, model)

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
        policy = mcts.search(neutral_state)
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
        if value == 1:
            print(player, "won")
        elif value == -1:
            print(player, "loss")
        elif value == 0:
            print("draw")
        print(get_board(state))
        # print number of black and white pieces
        print("black:", np.sum(state == 1))
        print("white:", np.sum(state == -1))
        break
    player = game.get_opponent(player)
