import numpy as np


class Othello:
    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.action_size = self.row_count * self.column_count

    def get_initial_state(self):
        # 0: empty, 1: black, -1: white
        init = np.zeros((self.row_count, self.column_count))
        init[3, 3] = -1
        init[3, 4] = 1
        init[4, 3] = 1
        init[4, 4] = -1
        return init

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        # flip opponent pieces
        for direction in range(8):
            if self.check_valid_move(state, row, column, direction, player):
                if direction == 0:
                    # up
                    for i in range(1, row + 1):
                        if state[row - i, column] == player:
                            break
                        state[row - i, column] = player
                if direction == 1:
                    # up-right
                    for i in range(1, min(row + 1, self.column_count - column)):
                        if state[row - i, column + i] == player:
                            break
                        state[row - i, column + i] = player
                if direction == 2:
                    # right
                    for i in range(column + 1, self.column_count):
                        if state[row, i] == player:
                            break
                        state[row, i] = player
                if direction == 3:
                    # down-right
                    for i in range(1, min(self.row_count - row, self.column_count - column)):
                        if state[row + i, column + i] == player:
                            break
                        state[row + i, column + i] = player
                if direction == 4:
                    # down
                    for i in range(row + 1, self.row_count):
                        if state[i, column] == player:
                            break
                        state[i, column] = player
                if direction == 5:
                    # down-left
                    for i in range(1, min(self.row_count - row, column + 1)):
                        if state[row + i, column - i] == player:
                            break
                        state[row + i, column - i] = player
                if direction == 6:
                    # left
                    for i in range(1, column + 1):
                        if state[row, column - i] == player:
                            break
                        state[row, column - i] = player
                if direction == 7:
                    # up-left
                    for i in range(1, min(row + 1, column + 1)):
                        if state[row - i, column - i] == player:
                            break
                        state[row - i, column - i] = player
        return state

    def check_valid_move(self, state, row, column, direction, player):
        # check if a move is valid in a given direction
        # direction: 0: up, 1: up-right, 2: right, 3: down-right, 4: down, 5: down-left, 6: left, 7: up-left
        # return True if valid, False if invalid
        # check if the move is on the board
        if row < 0 or row >= self.row_count or column < 0 or column >= self.column_count:
            return False

        # # check if the move is on an empty space
        # if state[row, column] != 0:
        #     return False
        # it is valid if there is at least one opponent piece that can be flipped and there is a player piece at the end
        if direction == 0:
            # up
            if row < 2:
                return False
            if state[row - 1, column] == -player:
                for i in range(2, row + 1):   #  i = 2, 3, ..., row
                    if state[row - i, column] == 0:
                        return False
                    if state[row - i, column] == player:
                        return True
            return False
        if direction == 1:
            # up-right
            if row < 2 or column >= self.column_count - 2:
                return False
            if state[row - 1, column + 1] == -player:
                for i in range(2, min(row + 1, self.column_count - column)):
                    if state[row - i, column + i] == 0:  
                        return False
                    if state[row - i, column + i] == player:
                        return True
            return False
        if direction == 2:
            # right
            if column >= self.column_count - 2:
                return False
            if state[row, column + 1] == -player:
                for i in range(column + 2, self.column_count):
                    if state[row, i] == 0:
                        return False
                    if state[row, i] == player:
                        return True
            return False
        if direction == 3:
            # down-right
            if row >= self.row_count - 2 or column >= self.column_count - 2:
                return False
            if state[row + 1, column + 1] == -player:
                for i in range(2, min(self.row_count - row, self.column_count - column)):
                    if state[row + i, column + i] == 0:
                        return False
                    if state[row + i, column + i] == player:
                        return True
            return False
        if direction == 4:
            # down
            if row >= self.row_count - 2:
                return False
            if state[row + 1, column] == -player:
                for i in range(row + 2, self.row_count):
                    if state[i, column] == 0:
                        return False
                    if state[i, column] == player:
                        return True
            return False
        if direction == 5:
            # down-left
            if row >= self.row_count - 2 or column < 2:
                return False
            if state[row + 1, column - 1] == -player:
                for i in range(2, min(self.row_count - row, column + 1)):
                    if state[row + i, column - i] == 0:
                        return False
                    if state[row + i, column - i] == player:
                        return True
            return False
        if direction == 6:
            # left
            if column < 2:
                return False
            if state[row, column - 1] == -player:
                for i in range(2, column + 1):
                    if state[row, column - i] == 0:
                        return False
                    if state[row, column - i] == player:
                        return True
            return False
        if direction == 7:
            # up-left
            if row < 2 or column < 2:
                return False
            if state[row - 1, column - 1] == -player:
                for i in range(2, min(row + 1, column + 1)):
                    if state[row - i, column - i] == 0:
                        return False
                    if state[row - i, column - i] == player:
                        return True
            return False

    def get_valid_moves(self, state, player=1):
        # always assume it is player 1 turn
        # return valid moves for player 1
        # it is a 1D array of size 64
        # 1: valid, 0: invalid
        valid_moves = np.zeros(self.action_size)
        for i in range(self.row_count):
            for j in range(self.column_count):
                if state[i, j] == 0:
                    # check if there is a valid move
                    # a valid move is a move that will flip at least one opponent piece
                    # in any direction
                    for direction in range(8):
                        if self.check_valid_move(state, i, j, direction, player):
                            valid_moves[i * self.column_count + j] = 1
                            break
        return valid_moves

    def check_win(self, state, action):
        if action == None:
            return 0, False

        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        # check if there is a valid move for opponent
        for i in range(self.row_count):
            for j in range(self.column_count):
                if state[i, j] == 0:
                    # check if there is a valid move
                    # a valid move is a move that will flip at least one opponent piece
                    # in any direction
                    for direction in range(8):
                        if self.check_valid_move(state, i, j, direction, -player):
                            return 0, False
        # if not then check if player has valid move
        for i in range(self.row_count):
            for j in range(self.column_count):
                if state[i, j] == 0:
                    # check if there is a valid move
                    # a valid move is a move that will flip at least one opponent piece
                    # in any direction
                    for direction in range(8):
                        if self.check_valid_move(state, i, j, direction, player):
                            return 0, False
        # if not then game is over
        # count pieces, whoever has more pieces wins
        player_pieces = np.sum(state == player)
        opponent_pieces = np.sum(state == -player)
        if player_pieces > opponent_pieces:
            return 1, True
        elif player_pieces < opponent_pieces:
            return -1, True
        else:
            return 0, True

    def get_value_and_terminated(self, state, action):
        return self.check_win(state, action)
        # if self.check_win(state, action):
        #     return 1, True
        # if np.sum(self.get_valid_moves(state)) == 0:
        #     return 0, True
        # return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state       
# tictactoe = Othello()
# player = 1
# 
# state = tictactoe.get_initial_state()
# 
# 
# while True:
#     print(state)
#     valid_moves = tictactoe.get_valid_moves(state, player)
#     print("valid_moves", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
#     action = int(input(f"{player}:"))
#     
#     if valid_moves[action] == 0:
#         print("action not valid")
#         continue
#         
#     state = tictactoe.get_next_state(state, action, player)
#     
#     value, is_terminal = tictactoe.get_value_and_terminated(state, action)
#     
#     if is_terminal:
#         print(state)
#         if value == 1:
#             print(player, "won")
#         else:
#             print("draw")
#         break
#         
#     player = tictactoe.get_opponent(player)

    
