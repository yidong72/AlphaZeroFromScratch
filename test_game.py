from othello_cython import Othello

game = Othello()
player = 1

state = game.get_initial_state()


while True:
    print(state)
    valid_moves = game.get_valid_moves(state, player)
    print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
    action = int(input(f"{player}:"))
    
    if valid_moves[action] == 0:
        print("action not valid")
        continue
        
    state = game.get_next_state(state, action, player)
    
    value, is_terminal = game.get_value_and_terminated(state, action)
    
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break
        
    player = game.get_opponent(player)

    
