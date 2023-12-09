import tkinter as tk
from PIL import ImageTk, Image
from game_engine import GameEngine
import time


class OthelloUI:
    def __init__(self, engine):
        self.engine = engine
        self.root = tk.Tk()
        self.root.title("Othello")

        # Create game board canvas
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack(pady=10)

        # Create buttons for starting and restarting the game
        self.start_button = tk.Button(self.root, text="Restart Game", command=self.start_game)
        self.start_button.pack(pady=5)

        # add a button to retract the last move
        self.retract_button = tk.Button(self.root, text="Retract Move", command=self.retract_game)
        self.retract_button.pack(pady=5)

        # Create labels for player turn and score
        self.player_turn_label = tk.Label(self.root, text="Player Turn: Black")
        self.player_turn_label.pack(pady=5)
        self.score_label = tk.Label(self.root, text="Black: 2 - White: 2")
        self.score_label.pack(pady=5)

        # create computer winning estimate label
        self.winner_estimate_label = tk.Label(self.root, text="Computer Winning Estimate: 0%")
        self.winner_estimate_label.pack(pady=5)

        self.previous_state = None

        # Bind mouse clicks to handle player moves
        self.canvas.bind("<Button-1>", self.on_click)

        # Initialize game state and images
        # self.images = {"black": ImageTk.PhotoImage(Image.open("black_piece.png")),
        #                "white": ImageTk.PhotoImage(Image.open("white_piece.png"))}
        self.start_game()

    def update_board(self):
        self.canvas.delete("all")
        # Draw squares and pieces based on current board state
        for y in range(8):
            for x in range(8):
                self.draw_square(x, y)
                if self.engine.state[y][x] != 0:
                    self.draw_piece(self.engine.state[y][x], x, y)
                else:
                    # Draw a dot on the square if it's a valid move
                    if self.engine.current_turn == 1:
                        if self.engine.is_human_move_valid(y, x):
                            self.canvas.create_oval(x * 50 + 20, y * 50 + 20, (x + 1) * 50 - 20, (y + 1) * 50 - 20,
                                                    fill="black")
        # refresh the canvas
        self.canvas.update()

    def raw_update_board(self, states):
        self.canvas.delete("all")
        # Draw squares and pieces based on current board state
        for y in range(8):
            for x in range(8):
                self.draw_square(x, y)
                if states[y][x] != 0:
                    self.draw_piece(states[y][x], x, y)
        # refresh the canvas
        self.canvas.update()

    def start_game(self):
        # Reset game state and UI elements
        self.engine.reset()
        self.player_turn_label.configure(text="Player Turn: Black")
        self.score_label.configure(text="Black: 2 - White: 2")
        self.update_board()
        self.previous_state = self.engine.state.copy()

    def retract_game(self):
        ### retract the last move
        if self.engine.current_turn == 1:
            self.engine.state = self.previous_state.copy()
            self.update_board()

    def on_click(self, event):
        # Get clicked square coordinates and convert to board indices
        x = event.x // 50
        # x is the column, y is the row
        y = event.y // 50
        # Check if clicked square is valid and make the move
        if not self.engine.is_human_move_valid(y, x):
            # Invalid move, show warning message
            self.player_turn_label.configure(text="Invalid Move!")
        else:
            # save the current state for retracting
            self.previous_state = self.engine.state.copy()
            self.player_turn_label.configure(text="")
            # place the piece and draw it
            copy_state = self.engine.state.copy()
            copy_state[y][x] = 1
            self.raw_update_board(copy_state)
            time.sleep(0.1)
            
        if self.engine.play_human_move(y, x):
            self.update_board()
            self.player_turn_label.configure(text=f"Computer is thinking...")
            # refresh the button to show the updated text
            self.canvas.update()
            if self.engine.can_computer_move():
                self.play_computer_move()
            else:
                if self.engine.is_game_over():
                    self.game_over_logics()
                else:
                    self.player_turn_label.configure(text="Computer cannot move! Still your move...")
                    self.engine.current_turn = 1

    def game_over_logics(self):
        black_score, white_score = self.engine.get_score()
        self.score_label.configure(text=f"Black: {black_score} - White: {white_score}")
        winner = "Black" if black_score > white_score else "White"
        if black_score == white_score:
            winner = "Draw"
        self.player_turn_label.configure(text=f"Game Over! Winner: {winner}")

    def play_computer_move(self):
        # Let AI make its move and update UI
        action, value = self.engine.play_computer_move()
        # update the winning estimate
        self.winner_estimate_label.configure(text=f"Computer Winning Estimate: {(value.item() + 1)/2.0  * 100:.2f}%")
        col = action % 8
        row = action // 8
        copy_state = self.engine.state.copy()
        copy_state[row][col] = -1
        self.raw_update_board(copy_state)
        time.sleep(0.1)
        self.engine.update_computer_board(action)
        self.player_turn_label.configure(text=f"")
        self.update_board()

        # Check game status and update score/labels
        if self.engine.is_game_over():
            self.game_over_logics()
        else:
            # Check if the user can make a move, if not, skip user turn and let AI play again
            if not self.engine.can_player_move():
                self.player_turn_label.configure(text="You cannot move! Skipping turn...")
                self.engine.current_turn = -1
                self.play_computer_move()
            else:
                self.player_turn_label.configure(text="Player Turn: Black")

    def draw_square(self, x, y):
        # Draw a square on the canvas based on its color
        color = "light gray" if (x + y) % 2 == 0 else "gray"
        self.canvas.create_rectangle(x * 50, y * 50, (x + 1) * 50, (y + 1) * 50, fill=color)

    def draw_piece(self, player, x, y):
        color = "black" if player == 1 else "white"
        # Place piece image on the canvas based on player and position
        margin = 5
        top_left = (x * 50 + margin, y * 50 + margin)
        bottom_right = ((x + 1) * 50 - margin, (y + 1) * 50 - margin)
        # Draw the piece
        self.canvas.create_oval(top_left, bottom_right, fill=color)


# Create and launch the UI
args = {
    'C': 2,
    'num_searches': 512,
    'dirichlet_epsilon': 0.0,
    'dirichlet_alpha': 0.01,
    'search': True,
    'temperature': 0,
}

engine = GameEngine("input_10.pt", args)
ui = OthelloUI(engine)
ui.root.mainloop()
