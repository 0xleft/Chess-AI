import random

import chess

from core.utils import *

from core.utils import decode_move

entropy = 0.5
DECAY = 0.99


def record_game(board, model):
    global entropy
    moves = []
    winner = None
    while True:
        if board.is_game_over():
            if board.result() == "1/2-1/2":
                board = chess.Board()
                moves = []
                board.push(list(board.legal_moves)[random.randint(0, len(list(board.legal_moves)) - 1)])
                continue
            break
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            # we never want to draw
            board = chess.Board()
            moves = []
        board_state = get_board_state(board)
        prediction = model.predict(board_state, verbose=0)
        from_move, from_num = get_move(prediction[0][:64], entropy)
        to_move, to_num = get_move(prediction[0][64:], entropy)
        if is_move_legal(board, from_move + to_move):
            entropy *= DECAY
            board.push_san(from_move + to_move)
            move = [0] * 128
            move[from_num] = 1
            move[to_num + 64] = 1
            moves.append([board_state, np.array([move])])
        else:
            first_legal_move = list(board.legal_moves)[random.randint(0, len(list(board.legal_moves)) - 1)]
            from_move_decoded, to_move_decoded = decode_move(first_legal_move)
            move = [0] * 128
            move[from_move_decoded] = 1
            move[to_move_decoded + 64] = 1
            board.push_san(str(first_legal_move))
            moves.append([board_state, np.array([move])])
            winner = "illegal"
            break

    if board.turn:
        return moves, winner
    else:
        return moves, winner


def get_move(input_matrix, entropy=0.5):
    move = np.argmax(input_matrix)
    out_move = NUMBER_TO_LETTER[move % 8] + str(move // 8 + 1)
    return out_move, move


def get_raw_move(move):
    return NUMBER_TO_LETTER[move % 8] + str(move // 8 + 1)