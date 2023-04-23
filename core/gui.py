import tkinter as tk

import chess
import numpy
from PIL import Image, ImageTk

from core.utils import get_board_state


class ChessGUI:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.root = tk.Tk()
        self.root.title("Chess")
        self.root.resizable(False, False)
        self.root.geometry(f"{self.width}x{self.height}")
        self.canvas = tk.Canvas(self.root, width=self.width / 2, height=self.height / 2, bg="white")
        self.canvas.pack()
        self.pieces = [
            "pieces/wp.png",
            "pieces/wn.png",
            "pieces/wb.png",
            "pieces/wr.png",
            "pieces/wq.png",
            "pieces/wk.png",
            "pieces/bp.png",
            "pieces/bn.png",
            "pieces/bb.png",
            "pieces/br.png",
            "pieces/bq.png",
            "pieces/bk.png"
        ]
        self.resize_images()

    def start(self):
        self.root.mainloop()

    def add_text(self, text, x, y):
        label = tk.Label(self.root, text=text)
        label.place(x=x, y=y)
        return label

    def add_button(self, text, command, x, y):
        button = tk.Button(self.root, text=text, command=command)
        button.place(x=x, y=y)
        return button

    def add_input(self, x, y):
        entry = tk.Entry(self.root)
        entry.place(x=x, y=y)
        return entry

    def add_dropdown_menu(self, options, x, y):
        variable = tk.StringVar(self.root)
        variable.set(options[0])
        menu = tk.OptionMenu(self.root, variable, *options)
        menu.place(x=x, y=y)
        return variable

    def resize_images(self):
        for index in range(len(self.pieces)):
            resized_image = Image.open(self.pieces[index]).resize(
                (int(self.canvas.winfo_width() * self.width / 16), int(self.canvas.winfo_height() * self.height / 16)),
                Image.ANTIALIAS)
            self.pieces[index] = ImageTk.PhotoImage(resized_image)

    def draw_square(self, x, y, color):
        self.canvas.create_rectangle(x * self.canvas.winfo_width() * self.width / 16,
                                     y * self.canvas.winfo_height() * self.height / 16,
                                     x * self.canvas.winfo_width() * self.width / 16 + self.canvas.winfo_width() * self.width / 16,
                                     y * self.canvas.winfo_height() * self.height / 16 + self.canvas.winfo_height() * self.height / 16,
                                     fill=color)

    def draw_board(self):
        for x in range(8):
            for y in range(8):
                if (x + y) % 2 == 0:
                    self.draw_square(x, y, "white")
                else:
                    self.draw_square(x, y, "gray")

    def draw_pieces(self, board):
        index = 0
        for peace_value in board[0, :64]:
            if peace_value == 0:
                index += 1
                continue
            if peace_value > 0:
                self.draw_piece(peace_value, "white", index)
            elif peace_value < 0:
                self.draw_piece(abs(peace_value), "black", index)
            index += 1

    def draw_piece(self, piece_type, color, index):
        image = self.pieces[(piece_type - 1) + 6 * int(color == "black")]
        self.canvas.create_image(
            (index % 8) * self.canvas.winfo_width() * self.width / 16 + self.canvas.winfo_width() * self.width / 32,
            (
                    index // 8) * self.canvas.winfo_height() * self.height / 16 + self.canvas.winfo_height() * self.height / 32,
            image=image)

    def draw(self, board: chess.Board):
        self.draw_board()
        self.draw_pieces(get_board_state(board))
        self.root.update()

    def draw_from_state(self, board_state: numpy.ndarray):
        self.draw_board()
        self.draw_pieces(board_state)
        self.canvas.update()
