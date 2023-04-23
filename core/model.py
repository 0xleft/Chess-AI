from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=65, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(128, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def train_model(input_data, model, chess_gui, epochs=1, verbose=0):
    for train_data in input_data:
        chess_gui.draw_from_state(train_data[0])
        model.fit(train_data[0], train_data[1], epochs=epochs, verbose=verbose)
    return model
