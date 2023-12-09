import tkinter as tk
from PIL import ImageTk, Image
from game_engine import GameEngine


class OthelloUI:
    def __init__(self, engine):
        self.engine = engine
        self.root = tk.Tk()
        self.root.title("Othello")

        # Create game board canvas
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack(pady=10)

        # Create buttons for starting and restarting the game
        self.start_button = tk.Button(self.root, text="Start Game", command=self.start_game)
        self.start_button.pack(pady=5)
        self.restart_button = tk.Button(self.root, text="Restart Game", command=self.restart_game, state="disabled")
        self.restart_button.pack(pady=5)

        # Create labels for player turn and score
        self.player_turn_label = tk.Label(self.root, text="Player Turn: Black")
        self.player_turn_label.pack(pady=5)
        self.score_label = tk.Label(self.root, text="Black: 2 - White: 2")
        self.score_label.pack(pady=5)

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
        # refresh the canvas
        self.canvas.update()

    def start_game(self):
        # Reset game state and UI elements
        self.engine.reset()
        self.restart_button.configure(state="disabled")
        self.player_turn_label.configure(text="Player Turn: Black")
        self.score_label.configure(text="Black: 2 - White: 2")
        self.update_board()

    def restart_game(self):
        self.start_game()

    def on_click(self, event):
        # Get clicked square coordinates and convert to board indices
        x = event.x // 50
        # x is the column, y is the row
        y = event.y // 50
        # Check if clicked square is valid and make the move
        if self.engine.play_human_move(y, x):
            self.update_board()
            self.play_computer_move()

    def play_computer_move(self):
        # Let AI make its move and update UI
        self.engine.play_computer_move()
        self.update_board()

        # Check game status and update score/labels
        if self.engine.is_game_over():
            black_score, white_score = self.engine.board.get_score()
            self.score_label.configure(text=f"Black: {black_score} - White: {white_score}")
            winner = "Black" if black_score > white_score else "White"
            if black_score == white_score:
                winner = "Draw"
            self.player_turn_label.configure(text=f"Game Over! Winner: {winner}")
            self.restart_button.configure(state="normal")

    def draw_square(self, x, y):
        # Draw a square on the canvas based on its color
        color = "light gray" if (x + y) % 2 == 0 else "gray"
        self.canvas.create_rectangle(x * 50, y * 50, (x + 1) * 50, (y + 1) * 50, fill=color)

    def draw_piece(self, player, x, y):
        color = "black" if player == 1 else "white"
        # Place piece image on the canvas based on player and position
        top_left = (x * 50, y * 50)
        bottom_right = ((x + 1) * 50, (y + 1) * 50)
        # Draw the piece
        self.canvas.create_oval(top_left, bottom_right, fill=color)


# Create and launch the UI
args = {
    'C': 2,
    'num_searches': 1024,
    'dirichlet_epsilon': 0.0,
    'dirichlet_alpha': 0.01,
    'search': True,
    'temperature': 0,
}

engine = GameEngine("input_10.pt", args)
ui = OthelloUI(engine)
ui.root.mainloop()
