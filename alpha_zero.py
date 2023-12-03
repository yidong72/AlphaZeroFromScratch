from mcts import MCTSParallel, SPG
import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import trange


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        

    def selfPlay(self):
        # for each game, start from the initial state
        # use the mcts to search for the best action
        # add the state, action_probs, and value to the memory
        # play the action and get the next state
        # switch player turn
        # repeat until the game is over
        # collect the memory from all the actual game play during the self play
        return_memory = []
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        players = np.array([1 for i in range(self.args['num_parallel_games'])])
        
        # play each of the spg to the end 
        count = 0
        while len(spGames) > 0:
            print(f'played {count} steps')
            # spg.state is on the perspective of player 1 or -1, not the neutral perspective
            states = np.stack([spg.state for spg in spGames])

            # handles the case where it need to skip a turn
            for n, spg in enumerate(spGames):
                player = players[n]
                valid_moves = self.game.get_valid_moves(spg.state, player)
                if np.sum(valid_moves) == 0:
                    # no moves for current player, skip turn
                    players[n] = self.game.get_opponent(player)

            # always use the neutral perspective for the mcts
            # states: (batch_size, row_count, column_count)
            # players: (batch_size)
            neutral_states = self.game.change_perspective(states, players[:, None, None])
            self.mcts.search(neutral_states, spGames)
            
            # loop from large to small so that we can remove spGames as we go
            for i in range(len(spGames))[::-1]:
                player = players[i]
                spg = spGames[i]
                
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                # the spg.root.state is the neutral state set at the beginning of the search 
                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs) # Divide temperature_action_probs with its sum in case of an error

                spg.state = self.game.get_next_state(spg.state, action, player)

                #  get the value from the perspective of player who just played `action`
                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    # loop through all the steps and add to the memory
                    # need to update the value based on the game play at the end of the games
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        # neurtal_state is on perspective of player 1
                        # memory always store the state from the perspective of player 1
                        # so need to change the value to the perspective of the neutral player
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]
                # switch player turn if the game is not over
                players[i] = self.game.get_opponent(player)

                    
                    
#            player = self.game.get_opponent(player)
            
        return return_memory
                
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")
            
