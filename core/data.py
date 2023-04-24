import random

import chess

from core.utils import *

from core.utils import decode_move

entropy = 0.5
DECAY = 0.999


def record_game(board, model):
    global entropy
    white_moves = []
    black_moves = []
    winner = None
    while True:
        board_state = get_board_state(board)
        prediction = model.predict(board_state, verbose=0)
        from_move, from_num = get_move(prediction[0][:64])
        to_move, to_num = get_move(prediction[0][64:])
        if board.is_repetition() or board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            white_moves, black_moves, winner = add_first_legal_move(white_moves, black_moves, board, board_state, shuffle=True)
            break
        if board.is_game_over():
            print("Game over")
            break
        if is_move_legal(board, from_move + to_move):
            entropy *= DECAY
            board.push_san(from_move + to_move)
            move = [0] * 128
            move[from_num] = 1
            move[to_num + 64] = 1
            if board.turn == chess.WHITE:
                winner = "white"
                white_moves.append([board_state, np.array([move])])
            elif board.turn == chess.BLACK:
                winner = "black"
                black_moves.append([board_state, np.array([move])])
        else:
            white_moves, black_moves, winner = add_first_legal_move(white_moves, black_moves, board, board_state)
            break

    if board.turn:
        return white_moves, winner
    else:
        return black_moves, winner


def add_first_legal_move(white_moves, black_moves, board, board_state, shuffle=True):
    legal_moves = list(board.legal_moves)
    if shuffle:
        random.shuffle(legal_moves)
    first_legal_move = legal_moves[0]
    from_move_decoded, to_move_decoded = decode_move(first_legal_move)
    move = [0] * 128
    move[from_move_decoded] = 1
    move[to_move_decoded + 64] = 1
    board.push_san(str(first_legal_move))
    if board.turn == chess.WHITE:
        white_moves.append([board_state, np.array([move])])
    elif board.turn == chess.BLACK:
        black_moves.append([board_state, np.array([move])])
    return white_moves, black_moves, "illegal"


def get_move(input_matrix):
    global entropy
    if random.random() < entropy:
        move = random.randint(0, 63)
        out_move = NUMBER_TO_LETTER[move % 8] + str(move // 8 + 1)
        return out_move, move
    move = np.argmax(input_matrix)
    out_move = NUMBER_TO_LETTER[move % 8] + str(move // 8 + 1)
    return out_move, move
