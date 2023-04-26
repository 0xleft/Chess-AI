import random

from keras import models

from core.data import get_move
from core.utils import *


def train(model, chess_gui, players=10):
    white_wins = 0
    black_wins = 0
    draws = 0
    illegal_moves = 0
    total_moves = 0
    max_moves = 0

    # create variations of model weights and create new models from them
    varied_models = []
    chess_gui.show_notification("Creating variations of model weights")
    for i in range(players):
        model_variation = variate_model_weights(model)
        varied_models.append(model_variation)
    varied_models.append(model)  # because it might still be the best model
    # play a game with each model
    random.shuffle(varied_models)
    while len(varied_models) != 1:
        result, moves = play_game(varied_models[0], varied_models[1], chess_gui)
        if result == "1-0":
            varied_models.pop(1)
            white_wins += 1
        elif result == "0-1":
            varied_models.pop(0)
            black_wins += 1
        else:
            draws += 1
            illegal_moves += 1
            varied_models.pop(random.randint(0, 1))

        total_moves += moves

    if len(varied_models) == 0:
        chess_gui.show_notification("No model won")
        return model.get_weights()

    chess_gui.show_notification("Model won")
    return varied_models[0].get_weights(), white_wins, black_wins, draws, illegal_moves, total_moves, max_moves


def variate_model_weights(model):
    # create copy of the model
    model_copy = models.clone_model(model)

    epsilon = 0.0001

    for layer in model_copy.layers:
        weights = layer.get_weights()
        for i in range(len(weights)):
            noise = np.random.uniform(low=-epsilon, high=epsilon, size=weights[i].shape)
            weights[i] += noise
        layer.set_weights(weights)

    return model_copy


def play_game(model1, model2, chess_gui):
    moves = 0
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move, prediction = get_best_move(board, model1)
        else:
            move, prediction = get_best_move(board, model2)
        chess_gui.draw_from_state(get_board_state(board))
        chess_gui.draw_move(prediction[0])
        moves += 1
        if not is_move_legal(board, move):
            return "illegal", moves
        board.push_san(move)
        chess_gui.draw_from_state(get_board_state(board))
        chess_gui.draw_move(move)
    return board.result(), moves


def get_best_move(board, model):
    board_state = get_board_state(board)
    prediction = model.predict(board_state, verbose=0)
    from_move, from_num = get_move(prediction[0][:64])
    to_move, to_num = get_move(prediction[0][64:])
    return from_move + to_move, prediction
