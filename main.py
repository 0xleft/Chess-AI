import os

import chess
from keras.saving.save import load_model, save_model

from board_generator import get_random_fen
from keras import Sequential, losses
from keras.layers import Dense
from keras.optimizers import Adam
from data import *
import chess_com_data
import matplotlib.pyplot as plt

neural_gui = NeuralGUI(800, 800)
chess_gui = ChessGUI(800, 800)


def create_model():
    model = Sequential()
    # input is 1x64
    model.add(Dense(64, input_dim=65, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # hidden layer 1x32
    model.add(Dense(128, activation='relu'))
    # hidden layer 1x256
    model.add(Dense(200, activation='relu'))
    # hidden layer 1x256
    model.add(Dense(200, activation='relu'))
    # hidden layer 1x256
    model.add(Dense(200, activation='relu'))
    # output is 1x64
    model.add(Dense(128, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model


def train_model(input_data, model):
    global counter
    for train_data in input_data:
        results = model.fit(train_data[0], train_data[1], epochs=1, verbose=0)
        # neural_gui.draw_neural_network(model)
        counter += 1
        if counter % 1000 == 0:
            save_model(model, f"models/model{counter}.h5")
    return model, results


counter = 0
if __name__ == '__main__':
    if os.path.exists("models/model.h5"):
        model = load_model("models/model.h5")
    else:
        print("Creating new model")
        model = create_model()

    print(model.summary())

    # for player in chess_com_data.get_players():
    #    print(player)
    #    for archive in chess_com_data.get_player_archives(player):
    #        print(archive)
    #        data = chess_com_data.collect_player_data(archive)
    #        model, results = train_model(data, model)
    #        save_model(model, "models/model.h5")

    while True:
        try:
            counter += 1
            if counter % 10000 == 0:
                save_model(model, f"models/model{counter}.h5")
            data = record_game(chess.Board(), model)
            model, results = train_model(data, model)
        except KeyboardInterrupt:
            save_model(model, "models/model.h5")
            exit(0)
