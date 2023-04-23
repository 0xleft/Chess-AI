import chess

from core import chess_com_data
from core.model import train_model
from data import record_game

training = False


def train_by_itself(model, chess_gui):
    global training
    training = True
    while training:
        data = record_game(chess.Board(), model)
        model = train_model(data, model, chess_gui)


def train_from_chess_com(model, chess_gui):
    global training
    training = True
    for player in chess_com_data.get_players():
        print(player)
        for archive in chess_com_data.get_player_archives(player):
            print(archive)
            if not training:
                return
            data = chess_com_data.collect_player_data(archive)
            model, results = train_model(data, model, chess_gui)


def stop_training():
    global training
    training = False


def set_training_mode(mode, model, chess_gui):
    if mode == "itself":
        train_by_itself(model, chess_gui)
    elif mode == "chesscom":
        train_from_chess_com(model, chess_gui)
