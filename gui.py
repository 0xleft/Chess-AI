import pygame
from keras.saving.save import save_model

import utils
import numpy as np


def get_board_state(board):
    board_state = utils.convert_to_int(board)
    board_state.append(int(board.turn))
    board_state = np.array(board_state)
    board_state = board_state.reshape(1, 65)
    return board_state


class ChessGUI:
    def __init__(self, width, height):
        self.screen = pygame.display.set_mode((width, height))
        self.width = width
        self.height = height
        pygame.init()
        self.font = pygame.font.SysFont("monospace", 15)
        self.clock = pygame.time.Clock()
        self.fps = 60

    def draw_text(self, text, color, x, y):
        self.screen.blit(self.font.render(text, True, color), (x, y))

    def draw_board(self, board, model):
        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                save_model(model, "models/model.h5")
                pygame.quit()
                quit()
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    pygame.draw.rect(self.screen, (255, 255, 255),
                                     (i * self.width / 8, j * self.height / 8, self.width / 8, self.height / 8))
                else:
                    pygame.draw.rect(self.screen, (0, 0, 0),
                                     (i * self.width / 8, j * self.height / 8, self.width / 8, self.height / 8))

        index = 0
        for peace_value in get_board_state(board)[0, :64]:
            if peace_value > 0:
                self.draw_piece(peace_value, "white", index)
            elif peace_value < 0:
                self.draw_piece(abs(peace_value), "black", index)
            index += 1

    def draw_piece(self, piece, color, index):
        if piece == 1:
            piece = "p"
        elif piece == 2:
            piece = "n"
        elif piece == 3:
            piece = "b"
        elif piece == 4:
            piece = "r"
        elif piece == 5:
            piece = "q"
        elif piece == 6:
            piece = "k"
        if color == "white":
            image = pygame.image.load("pieces/w" + piece + ".png")
        else:
            image = pygame.image.load("pieces/b" + piece + ".png")
        image = pygame.transform.scale(image, (int(self.width / 8), int(self.height / 8)))
        self.screen.blit(image, (index % 8 * self.width / 8, index // 8 * self.height / 8))

    def update(self):
        pygame.display.update()

    def clear(self):
        self.screen.fill((0, 0, 0))
