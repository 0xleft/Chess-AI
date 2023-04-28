import math

import numpy
import numpy as np
import requests
from core.utils import *


def decode_tcn(tcn):
    T = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?{~}(^)[_]@#$,./&-*++="

    length = len(tcn)
    output = []
    for c in range(0, length, 2):
        d = {}
        start = T.index(tcn[c])
        end = T.index(tcn[c + 1])
        if end > 63:
            d["promotion"] = T[math.floor((end - 64) / 3)]
            end = (end - 64) % 3 + 56
        if start > 75:
            d["drop"] = T[start - 79]
        else:
            d["from"] = T[start % 8] + str((math.floor(start / 8) + 1))
        d["to"] = T[end % 8] + str((math.floor(end / 8) + 1))
        output.append(d)
    return output


def get_players():
    url = "https://api.chess.com/pub/titled/GM"
    response = requests.get(url)
    data = response.json()
    players = data["players"]
    return players


def get_player_archives(player):
    url = "https://api.chess.com/pub/player/" + player + "/games/archives"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    data = response.json()
    archives = data["archives"]
    return archives


def collect_player_data(url):
    game_data = []
    response = requests.get(url)
    data = response.json()
    games = data["games"]
    for game in games:
        board = chess.Board()
        tnc = game["tcn"]
        decoded_tnc = decode_tcn(tnc)
        try:
            for move in decoded_tnc:
                from_num, to_num = decode_move(move["from"] + move["to"])
                move_out = [0] * 128
                move_out[from_num] = 1
                move_out[to_num + 64] = 1
                game_data.append([get_board_state(board), np.array([move_out]), board.fen()])
                if "drop" in move:
                    board.push_san(move["drop"] + "@" + move["to"])
                else:
                    board.push_san(move["from"] + move["to"])
        except chess.IllegalMoveError:
            continue
        except chess.InvalidMoveError:
            continue
    print("Collected " + str(len(game_data)) + " moves")
    return game_data


def collect_player_data_moves(url):
    game_data = []
    response = requests.get(url)
    data = response.json()
    games = data["games"]
    for game in games:
        board = chess.Board()
        tnc = game["tcn"]
        decoded_tnc = decode_tcn(tnc)
        game_moves = []
        try:
            for move in decoded_tnc:
                move = move["from"] + move["to"]

                from_move_index, to_move_index = decode_move(move)

                move_matrix = np.zeros((2, 64))

                move_matrix[0][from_move_index] = 1
                move_matrix[1][to_move_index] = 1

                input_data = np.full(shape=(80, 2), fill_value=numpy.array((65, 65)))

                for i in range(len(game_moves)):
                    input_data[i] = game_moves[i]

                game_data.append(
                    [
                        input_data,  # 1x80
                        move_matrix  # 2x64
                    ]
                )

                game_moves.append((from_move_index, to_move_index))
                board.push_san(move)
        except chess.IllegalMoveError:
            continue
        except chess.InvalidMoveError:
            continue
        except IndexError:
            continue
    print("Collected " + str(len(game_data)) + " moves")
    return game_data

