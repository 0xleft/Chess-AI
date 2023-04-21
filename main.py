import os

import chess
from keras.saving.save import load_model, save_model

from board_generator import get_random_fen
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from data import *


def create_model():
    model = Sequential()
    # input is 1x64
    model.add(Dense(64, input_dim=65, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # hidden layer 1x32
    model.add(Dense(32, activation='relu'))
    # hidden layer 1x128
    model.add(Dense(128, activation='relu'))
    # hidden layer 1x256
    model.add(Dense(256, activation='relu'))
    # hidden layer 1x500
    model.add(Dense(500, activation='relu'))
    # hidden layer 1x256
    model.add(Dense(256, activation='relu'))
    # hidden layer 1x128
    model.add(Dense(200, activation='relu'))
    # output is 1x64
    model.add(Dense(128, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00001), metrics=['accuracy'])
    return model


def train_model(input_data, model):
    for train_data in input_data:
        model.fit(train_data[0], train_data[1], epochs=1, verbose=0)
    return model


counter = 0
if __name__ == '__main__':
    if os.path.exists("models/model.h5"):
        model = load_model("models/model.h5")
    else:
        model = create_model()
    while True:
        try:
            counter += 1
            if counter % 10000 == 0:
                save_model(model, f"models/model{counter}.h5")
            data = record_game(chess.Board(), model)
            model = train_model(data, model)
        except KeyboardInterrupt:
            save_model(model, "models/model.h5")
            exit(0)
