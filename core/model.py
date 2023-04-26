import chess
import keras.saving.save
from keras import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam

from core import chess_com_data, genetic_algorithm
from core.data import record_game, get_move
from core.utils import get_board_state, decode_move
from keras.saving.save import save_model

training = False


def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=65, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(210, activation='linear'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='linear'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='linear'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(500, activation='linear'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='linear'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(128, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.000001), metrics=['accuracy'])
    return model


def train_model(input_data, model, chess_gui, epochs=1, verbose=0):
    results = None
    for train_data in input_data:
        if not training:
            break
        chess_gui.draw_from_state(train_data[0])
        # draw move ---
        move = train_data[1][0]
        chess_gui.draw_move(move)
        results = model.fit(train_data[0], train_data[1], epochs=epochs, verbose=verbose)
    if results is None:
        results = {"loss": [0], "accuracy": [0]}
    return model, results


def stop_training():
    global training
    training = False


def train_by_itself(model, chess_gui):
    total_moves = 0
    white_wins = 0
    black_wins = 0
    illegal_moves = 0
    max_moves = 0
    legal_moves = 0
    global training
    chess_gui.show_notification("Starting training")
    training = True
    while training:
        data, winner = record_game(chess.Board(), model)
        total_moves += len(data)
        chess_gui.update_statistic("total_moves", f"Total moves: {total_moves}")
        if winner == "white":
            chess_gui.update_statistic("white_wins", f"White wins: {white_wins}")
            white_wins += 1
        elif winner == "black":
            chess_gui.update_statistic("black_wins", f"Black wins: {black_wins}")
            black_wins += 1
        elif winner == "draw":
            pass
        elif winner == "illegal":
            chess_gui.update_statistic("illegal_moves", f"Illegal moves: {illegal_moves}")
            illegal_moves += 1

        legal_moves_done = len(data) - 1
        if winner != "illegal":
            legal_moves_done = len(data)
        legal_moves += legal_moves_done
        if legal_moves_done > max_moves:
            max_moves = legal_moves_done
        chess_gui.update_statistic("max_moves", f"Max moves: {max_moves}")
        chess_gui.update_statistic("legal_moves", f"Legal moves: {legal_moves}")

        if (total_moves % 20000) == 0:
            chess_gui.show_notification(f"Total moves: {total_moves}")
            model.save_weights(f"models/model{total_moves}_weights.h5")

        chess_gui.update_statistic("move_ratio",
                                   f"Legal/Illegal move ratio: {round(legal_moves / total_moves, 4)} : {round(illegal_moves / total_moves, 4)}")
        model, results = train_model(data, model, chess_gui)
        loss = results.history['loss'][0]
        chess_gui.update_statistic("loss", f"Loss: {loss}")


def train_from_chess_com(model, chess_gui):
    total_moves = 0
    chess_gui.show_notification("Starting training")
    global training
    training = True
    for player in chess_com_data.get_players():
        print(player)
        chess_gui.show_notification("Downloading " + player)
        for archive in chess_com_data.get_player_archives(player):
            print(archive)
            if not training:
                return
            chess_gui.show_notification("Training from " + archive, 10000)
            data = chess_com_data.collect_player_data(archive)
            model, results = train_model(data, model, chess_gui)
            loss = results.history['loss'][0]
            chess_gui.update_statistic("loss", f"Loss: {loss}")
            total_moves += len(data)
            chess_gui.update_statistic("total_moves", f"Total moves: {total_moves}")
            if (total_moves % 20000) == 0:
                chess_gui.show_notification(f"Total moves: {total_moves}")
                save_model(model, f"models/model{total_moves}_chesscom.h5")


def train_special_mode(model, chess_gui, player_name):
    chess_gui.show_notification("Starting training")
    global training
    training = True
    for archive in chess_com_data.get_player_archives(player_name):
        print(archive)
        if not training:
            return
        chess_gui.show_notification("Training from " + archive, 10000)
        data = chess_com_data.collect_player_data(archive)
        model = train_model(data, model, chess_gui)


def train_genetic_algorithm(model, chess_gui):
    chess_gui.show_notification("Starting genetic training")
    max_moves = 0
    illegal_moves = 0
    global training
    training = True
    while training:
        if not training:
            break
        new_weights = genetic_algorithm.train(model, chess_gui)[0]
        model.set_weights(new_weights)


def set_training_mode(mode, model, chess_gui):
    if training:
        chess_gui.show_notification("Already training")
        return
    if mode == "itself":
        train_by_itself(model, chess_gui)
    elif mode == "chesscom":
        train_from_chess_com(model, chess_gui)
    elif mode == "genetic":
        train_genetic_algorithm(model, chess_gui)
    else:
        train_special_mode(model, chess_gui, mode)


def test_predict(chess_gui, model, fen):
    chess_board = chess.Board()
    try:
        chess_board = chess.Board(fen)
    except ValueError:
        print("Invalid fen, defaulting")
        chess_gui.show_notification("Invalid fen, defaulting")
    if training:
        return
    chess_gui.draw(chess_board)
    board_state = get_board_state(chess_board)
    prediction = model.predict(board_state, verbose=0)

    _, from_num = get_move(prediction[0][:64], 0)
    _, to_num = get_move(prediction[0][64:], 0)

    from_x = from_num % 8
    from_y = from_num // 8
    to_x = to_num % 8
    to_y = to_num // 8
    chess_gui.draw_arrow(from_x, from_y, to_x, to_y, "red")
