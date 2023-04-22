import random

from gui import *
from constants import *
import chess
from utils import *

from utils import decode_move

gui = ChessGUI(500, 500)
highest = 0
entropy = 0.5
DECAY = 0.99999999999


def record_game(board, model):
    global highest, entropy
    white_moves = []
    black_moves = []
    while True:
        if board.is_game_over():
            break
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            break
        board_state = get_board_state(board)
        prediction = model.predict(board_state, verbose=0)
        from_move, from_num = get_move(prediction[0][:64])
        to_move, to_num = get_move(prediction[0][64:])
        if is_move_legal(board, from_move + to_move):
            entropy *= DECAY
            board.push_san(from_move + to_move)
            move = [0] * 128
            move[from_num] = 1
            move[to_num + 64] = 1
            if board.turn:
                white_moves.append([board_state, np.array([move])])
            else:
                black_moves.append([board_state, np.array([move])])
            gui.clear()
            gui.draw_board(get_board_state(board), model)
            if len(white_moves) + len(black_moves) > highest:
                highest = len(white_moves) + len(black_moves)
            gui.draw_text(f"Most moves: {highest} \n Last move: {from_move + to_move} \n Entropy: {entropy} \n",
                          (255, 0, 0), 0, 0)
            gui.update()
        else:
            first_legal_move = list(board.legal_moves)[random.randint(0, len(list(board.legal_moves)) - 1)]
            from_move_decoded, to_move_decoded = decode_move(first_legal_move)
            move = [0] * 128
            move[from_move_decoded] = 1
            move[to_move_decoded + 64] = 1
            board.push_san(str(first_legal_move))
            gui.clear()
            gui.draw_board(get_board_state(board), model)
            if len(white_moves) + len(black_moves) > highest:
                highest = len(white_moves) + len(black_moves)
            gui.draw_text(f"Most moves: {highest} \n Last move: {from_move + to_move} \n Entropy: {entropy} \n",
                          (255, 0, 0), 0, 0)
            gui.update()
            # this is just so it learns to not make illegal moves
            if board.turn:
                white_moves.append([board_state, np.array([move])])
            else:
                black_moves.append([board_state, np.array([move])])
            break

    if board.turn:
        return white_moves
    else:
        return black_moves


def get_move(input_matrix):
    global entropy
    if random.random() < entropy:
        move = random.randint(0, 63)
        out_move = NUMBER_TO_LETTER[move % 8] + str(move // 8 + 1)
        return out_move, move
    move = np.argmax(input_matrix)
    out_move = NUMBER_TO_LETTER[move % 8] + str(move // 8 + 1)
    return out_move, move
