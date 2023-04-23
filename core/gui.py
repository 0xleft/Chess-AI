import tkinter as tk

import chess
import numpy
from PIL import Image, ImageTk
from core.utils import get_board_state

from keras_visualizer import visualizer
import os


def open_graph(event):
    os.system("graph.png")


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
        self.neural_network_canvas = tk.Canvas(self.root, width=self.width / 2, height=self.height / 2, bg="white")
        self.neural_network_canvas.pack()
        self.statistics = {}
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

    def add_statistic(self, id, placeholder, x, y):
        self.statistics[id] = (self.add_text(placeholder, x, y))

    def update_statistic(self, id, value):
        self.statistics[id].config(text=value)
        self.root.update()

    def draw_arrow(self, from_x, from_y, to_x, to_y, color):
        self.canvas.create_line(from_x * self.canvas.winfo_width() / 8 + self.canvas.winfo_width() / 16,
                                from_y * self.canvas.winfo_height() / 8 + self.canvas.winfo_height() / 16,
                                to_x * self.canvas.winfo_width() / 8 + self.canvas.winfo_width() / 16,
                                to_y * self.canvas.winfo_height() / 8 + self.canvas.winfo_height() / 16,
                                arrow=tk.LAST, fill=color)

    def draw_square(self, x, y, color):
        self.canvas.create_rectangle(x * self.canvas.winfo_width() / 8,
                                     y * self.canvas.winfo_height() / 8,
                                     x * self.canvas.winfo_width() / 8 + self.canvas.winfo_width() / 8,
                                     y * self.canvas.winfo_height() / 8 + self.canvas.winfo_height() / 8,
                                     fill=color)

    def draw_board(self):
        self.canvas.delete("all")
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
            (index % 8) * self.canvas.winfo_width() / 8 + self.canvas.winfo_width() / 16,
            (index // 8) * self.canvas.winfo_height() / 8 + self.canvas.winfo_height() / 16,
            image=image
        )

    def add_container(self, x, y):
        container = tk.Frame(self.root)
        container.place(x=x, y=y)
        return container

    def draw(self, board: chess.Board):
        self.draw_board()
        self.draw_pieces(get_board_state(board))
        self.root.update()

    def draw_from_state(self, board_state: numpy.ndarray):
        self.draw_board()
        self.draw_pieces(board_state)
        self.canvas.update()

    def show_notification(self, text, time=5000):
        text = self.add_text(text, self.width / 2, self.height / 2)
        self.root.after(time, lambda: text.destroy())
        self.root.update()

    def draw_neural_network(self, model):
        self.neural_network_canvas.delete("all")
        visualizer(model, file_name='graph', file_format="png", view=False, settings=None)
        image = Image.open("graph.png").resize(
            (int(self.neural_network_canvas.winfo_width()), int(self.neural_network_canvas.winfo_height())),
            Image.ANTIALIAS)
        self.neural_network_canvas.image = ImageTk.PhotoImage(image)
        self.neural_network_canvas.create_image(0, 0, image=self.neural_network_canvas.image, anchor='nw')
        self.neural_network_canvas.bind("<Button-1>", open_graph)
        self.neural_network_canvas.update()

