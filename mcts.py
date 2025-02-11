import numpy as np
import math
import torch
from typing import List


class SPG:
    """a class to store the state, root node, and current node of a single player game
    """
    def __init__(self, game):
        self.state = game.get_initial_state()
        # memory is a list of (state, improved policy, player) tuples
        # from the root state to the end of the game
        self.memory = []
        self.root = None
        self.node = None


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        # whether the parent turn is skipped or not
        self.skip_parent = False
    
    def __repr__(self) -> str:
        # balck #, white o, empty .
        output = ''
        for i in range(self.game.row_count):
            for j in range(self.game.column_count):
                if self.state[i][j] == 1:
                    output += '# '
                elif self.state[i][j] == -1:
                    output += 'o '
                else:
                    output += '. '
            output += '\n'
        return output
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        # child value_sum has different meanining from the parent value, beacuse the child value_sum is from the opponent's perspective
        # exception, if the parent turn is skipped, then the child value_sum is from the perspective of the current player
        if child.visit_count == 0:
            q_value = 0.5 # 0.5 is the prior that it can win or loss half of the time
        else:
            if not child.skip_parent:
                q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
            else:
                q_value = (child.value_sum / child.visit_count + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        if not self.skip_parent:
            # only flip the value if the parent turn is not skipped
            value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search_all(self, state):
        root = Node(self.game, self.args, state, visit_count=1)
        
        policy, return_value = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()
            # value is from the perspective of the parent node (opponent piece), who just play the node.action_taken action.
            # the piece on node.action_taken is the opponent's piece, 
            # because the state is neutral perspective, so the player -1 takes the action node.action_taken
            # value is 1 if the opponent wins, -1 if the opponent loses, 0 if draw
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            # the value from the perspective of the current player, need to flip the sign of the value 
            # exception, if the parent turn is skipped, then the value is from the perspective of the current player
            if not node.skip_parent:
                value = self.game.get_opponent_value(value)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                if np.sum(valid_moves) == 0:
                    # if no valid moves, change current node to the opponent's perspective
                    # flip the state and the value
                    node.state = self.game.change_perspective(node.state, player=-1)
                    node.value_sum = -node.value_sum
                    node.skip_parent = True
                    continue
                else:
                    policy *= valid_moves
                    policy /= np.sum(policy)
                    value = value.item()
                    node.expand(policy)

            node.backpropagate(value)


        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs, return_value

    def search(self, state):
        policy, _ = self.search_all(state)
        return policy



class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, states, spGames: List[SPG]):
        # states: (batch_size, row_count, column_count), neutral perspective
        # spGames: list of spGame
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        # policy: (batch_size, action_size)
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)
            # no need to handle the case that no valid moves
            # because the we search the states[i] which has at least one valid move
            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)
        
        for search in range(self.args['num_searches']):
            # consider doing this in parallel
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()
                
                # check the move done (node.action_taken) by the opponent resulting in game over or not
                # also get the value from the perspective of the opponent
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                if not node.skip_parent:
                    value = self.game.get_opponent_value(value)

                if is_terminal:
                    # if terminal, then backpropagate the value, and skip the expansion of the node because spg.node is None
                    node.backpropagate(value)
                else:
                    # if not terminal, then expand the node in the later part of the code
                    spg.node = node
                    
            # index of spGames that are expandable
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
                    
            if len(expandable_spGames) > 0:
                # compute the batched policy and value for the expandable spGames

                # get the states of the expandable spGames
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                # policy: (batch_size, action_size)
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                # value: (batch_size, 1)
                value = value.cpu().numpy()
                
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                
                # valid move is from the perspective of the current player (playe 1)
                valid_moves = self.game.get_valid_moves(node.state)
                if np.sum(valid_moves) == 0:
                    # if no valid moves, change current node to the opponent's perspective, no need to expand
                    # flip the state and the value
                    node.state = self.game.change_perspective(node.state, player=-1)
                    node.value_sum = -node.value_sum
                    node.skip_parent = True
                    continue
                else:
                    spg_policy *= valid_moves
                    spg_policy /= np.sum(spg_policy)
                    node.expand(spg_policy)

                node.backpropagate(spg_value)
