import random

from gui import *
from constants import *
import chess


gui = ChessGUI(500, 500)


def record_game(board, model):
    white_moves = []
    black_moves = []
    while True:
        if board.is_game_over():
            break
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            board = chess.Board()
            white_moves = []
            black_moves = []
        board_state = get_board_state(board)
        prediction = model.predict(board_state, verbose=0)
        from_move, from_num = get_move(prediction[0][:64])
        to_move, to_num = get_move(prediction[0][64:])
        if is_legal_move(from_move + to_move, board):
            board.push_san(from_move + to_move)
            move = [0] * 128
            move[from_num] = 1
            move[to_num + 64] = 1

            if board.turn:
                white_moves.append([board_state, np.array([move])])
            else:
                black_moves.append([board_state, np.array([move])])
            gui.clear()
            gui.draw_board(board, model)
            gui.draw_text("Moves amount: " + str(len(white_moves) + len(black_moves)), (255, 0, 0), 0, 0)
            gui.update()
        else:
            gui.clear()
            gui.draw_board(board, model)
            gui.draw_text("Moves amount: " + str(len(white_moves) + len(black_moves)), (255, 0, 0), 0, 0)
            gui.update()
            first_legal_move = list(board.legal_moves)[random.randint(0, len(list(board.legal_moves)) - 1)]
            from_move_decoded, to_move_decoded = decode_move(first_legal_move)
            move = [0] * 128
            move[from_move_decoded] = 1
            move[to_move_decoded + 64] = 1
            if board.turn:
                white_moves.append([board_state, np.array([move])])
            else:
                black_moves.append([board_state, np.array([move])])
            break

    if board.turn:
        return white_moves
    else:
        return black_moves


def is_legal_move(move, board):
    try:
        board.push_san(move)
        board.pop()
        return True
    except chess.IllegalMoveError:
        return False
    except chess.InvalidMoveError:
        return False


def decode_move(move):
    characters = str(move)
    from_move_column = characters[0]
    from_move_row = characters[1]
    to_move_column = characters[2]
    to_move_row = characters[3]

    from_move = LETTER_TO_NUMBER[from_move_column] + (int(from_move_row) - 1) * 8
    to_move = LETTER_TO_NUMBER[to_move_column] + (int(to_move_row) - 1) * 8

    return from_move, to_move


def get_move(input_matrix):
    move = np.argmax(input_matrix)
    out_move = NUMBER_TO_LETTER[move % 8] + str(move // 8 + 1)
    return out_move, move
