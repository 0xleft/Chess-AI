import chess
from constants import *


def convert_to_int(board):
    board_state = []
    for i in range(64):
        if board.piece_at(i) is None:
            board_state.append(0)
        else:
            if board.piece_at(i).color == chess.WHITE:
                board_state.append(board.piece_at(i).piece_type)
            else:
                board_state.append(-board.piece_at(i).piece_type)
    # flip vertically
    board_state = board_state[::-1]
    # flip horizontally
    board_state = [board_state[i:i + 8][::-1] for i in range(0, 64, 8)]
    board_state = [item for sublist in board_state for item in sublist]
    return board_state


def decode_move(move):
    characters = str(move)
    from_move_column = characters[0]
    from_move_row = characters[1]
    to_move_column = characters[2]
    to_move_row = characters[3]

    from_move = LETTER_TO_NUMBER[from_move_column] + (int(from_move_row) - 1) * 8
    to_move = LETTER_TO_NUMBER[to_move_column] + (int(to_move_row) - 1) * 8

    return from_move, to_move


def is_move_legal(board, move):
    for legal_move in board.legal_moves:
        if legal_move == move:
            return True
    return False
