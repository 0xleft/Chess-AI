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


class NeuralGUI:
    def __init__(self, width, height):
        self.screen = pygame.display.set_mode((width, height))
        self.width = width
        self.height = height
        pygame.init()
        self.font = pygame.font.SysFont("monospace", 15)
        self.clock = pygame.time.Clock()
        self.fps = 60

    def draw_neural_network(self, model):
        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                save_model(model, "models/model.h5")
                pygame.quit()
                quit()
        self.screen.fill((0, 0, 0))
        index = 0
        for layer in model.layers:
            self.draw_layer(layer, index, model)
            index += 1
        pygame.display.update()

    def draw_layer(self, layer, index, model):
        if index == 0:
            self.draw_input_layer(layer, index)
        else:
            self.draw_hidden_layer(layer, index)

    def draw_input_layer(self, layer, index):
        neuron_index = 0
        for neuron in range(layer.get_config()['batch_input_shape'][1]):
            self.draw_neuron(neuron_index, index)
            self.draw_weights(neuron_index, index, layer.get_weights()[0])
            neuron_index += 1

    def draw_hidden_layer(self, layer, index):
        neuron_index = 0
        for neuron in range(layer.get_config()['units']):
            self.draw_neuron(neuron_index, index)
            self.draw_weights(neuron_index, index, layer.get_weights()[0])
            neuron_index += 1

    def draw_weights(self, neuron_index, index, weights):
        weight_index = 0
        for weight in weights[0]:
            out_weight = abs(weight + 1) * 255
            if weight < 0:
                out_weight = 255 - out_weight
            try:
                pygame.draw.line(self.screen, color=(out_weight, out_weight, out_weight), width=1,
                                 start_pos=(index * 100 + 10, neuron_index * 3 + 10),
                                 end_pos=((index - 1) * 100 + 10, weight_index * 3 + 10))
            except ValueError:
                pass
            weight_index += 1

    def draw_neuron(self, neuron_index, index):
        pygame.draw.circle(self.screen, color=(255, 255, 255), radius=1, width=1,
                           center=(index * 100 + 10, neuron_index * 3 + 10))


class ChessGUI:
    def __init__(self, width, height):
        self.screen = pygame.display.set_mode((width, height))
        self.width = width
        self.height = height
        pygame.init()
        self.font = pygame.font.SysFont("monospace", 15)
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.black_rook = pygame.transform.scale(pygame.image.load("pieces/br.png"),
                                                 (int(self.width / 8), int(self.height / 8)))
        self.black_knight = pygame.transform.scale(pygame.image.load("pieces/bn.png"),
                                                   (int(self.width / 8), int(self.height / 8)))
        self.black_bishop = pygame.transform.scale(pygame.image.load("pieces/bb.png"),
                                                   (int(self.width / 8), int(self.height / 8)))
        self.black_queen = pygame.transform.scale(pygame.image.load("pieces/bq.png"),
                                                  (int(self.width / 8), int(self.height / 8)))
        self.black_king = pygame.transform.scale(pygame.image.load("pieces/bk.png"),
                                                 (int(self.width / 8), int(self.height / 8)))
        self.black_pawn = pygame.transform.scale(pygame.image.load("pieces/bp.png"),
                                                 (int(self.width / 8), int(self.height / 8)))
        self.white_rook = pygame.transform.scale(pygame.image.load("pieces/wr.png"),
                                                 (int(self.width / 8), int(self.height / 8)))
        self.white_knight = pygame.transform.scale(pygame.image.load("pieces/wn.png"),
                                                   (int(self.width / 8), int(self.height / 8)))
        self.white_bishop = pygame.transform.scale(pygame.image.load("pieces/wb.png"),
                                                   (int(self.width / 8), int(self.height / 8)))
        self.white_queen = pygame.transform.scale(pygame.image.load("pieces/wq.png"),
                                                  (int(self.width / 8), int(self.height / 8)))
        self.white_king = pygame.transform.scale(pygame.image.load("pieces/wk.png"),
                                                 (int(self.width / 8), int(self.height / 8)))
        self.white_pawn = pygame.transform.scale(pygame.image.load("pieces/wp.png"),
                                                 (int(self.width / 8), int(self.height / 8)))
        self.prediction = pygame.transform.scale(pygame.image.load("pieces/prediction.png"),
                                                 (int(self.width / 8), int(self.height / 8)))

    def update_board(self, board, model):
        self.clear()
        self.draw_board(get_board_state(board), model)
        self.update()

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
        for peace_value in board[0, :64]:
            if peace_value > 0:
                self.draw_piece(peace_value, "white", index)
            elif peace_value < 0:
                self.draw_piece(abs(peace_value), "black", index)
            index += 1

    def draw_piece(self, piece, color, index):
        image = None
        if color == "white":
            if piece == 1:
                image = self.white_pawn
            elif piece == 2:
                image = self.white_knight
            elif piece == 3:
                image = self.white_bishop
            elif piece == 4:
                image = self.white_rook
            elif piece == 5:
                image = self.white_queen
            elif piece == 6:
                image = self.white_king
        elif color == "prediction":
            if piece == 0:
                image = self.prediction
            if piece == 1:
                image = self.prediction
        else:
            if piece == 1:
                image = self.black_pawn
            elif piece == 2:
                image = self.black_knight
            elif piece == 3:
                image = self.black_bishop
            elif piece == 4:
                image = self.black_rook
            elif piece == 5:
                image = self.black_queen
            elif piece == 6:
                image = self.black_king
        self.screen.blit(image, (index % 8 * self.width / 8, index // 8 * self.height / 8))

    def update(self):
        pygame.display.update()

    def clear(self):
        self.screen.fill((0, 0, 0))