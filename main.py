from datetime import datetime
import os

from core.gui import ChessGUI
from keras.saving.save import load_model, save_model
from core import model
from core.model import set_training_mode, stop_training, create_model, test_predict
import chess

def load_model_weights(model_name, model, chess_gui):
    print(model.summary())
    chess_gui.draw_neural_network(model)
    try:
        model.load_weights(model_name)
    except IOError:
        chess_gui.show_notification("Model " + model_name + " not found")
        return
    chess_gui.show_notification("Loaded model weights " + model_name)


if __name__ == '__main__':
    chess_gui = ChessGUI(1000, 1000)
    print("Creating new model")
    model = create_model()
    chess_gui.show_notification("Created new model")
    if os.path.exists("models/model_weights.h5"):
        model.load_weights("models/model_weights.h5")
    else:
        chess_gui.show_notification("No model weights found")

    print(model.summary())
    board = chess.Board()

    training_options = chess_gui.add_dropdown_menu(["itself", "chesscom"], 100, 0)
    start_training_button = chess_gui.add_button("Start training", lambda: set_training_mode(training_options.get(), model, chess_gui), 0, 0)
    stop_training_button = chess_gui.add_button("Stop training", lambda: stop_training(), 0, 25)
    save_model_button = chess_gui.add_button("Save model", lambda: model.save_weights(f"models/model{str(datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')}_weights.h5"), 0, 50)

    # statistics
    chess_gui.add_statistic("white_wins", "White wins: 0", 0, 100)
    chess_gui.add_statistic("black_wins", "Black wins: 0", 0, 125)
    chess_gui.add_statistic("max_moves", "Max moves: 0", 0, 175)
    chess_gui.add_statistic("illegal_moves", "Illegal moves: 0", 0, 200)
    chess_gui.add_statistic("legal_moves", "Legal moves: 0", 0, 225)
    chess_gui.add_statistic("total_moves", "Total moves: 0", 0, 250)
    chess_gui.add_statistic("loss", "Loss: 0", 0, 275)

    board_fen_input = chess_gui.add_input(0, 500)
    board_fen_button = chess_gui.add_button("Test prediction", lambda: test_predict(chess_gui, model, board_fen_input.get()), 0, 525)

    special_training_input = chess_gui.add_input(0, 600)
    special_training_button = chess_gui.add_button("Special training", lambda: set_training_mode(special_training_input.get(), model, chess_gui), 0, 625)

    model_name_input = chess_gui.add_input(0, 700)
    model_name_button = chess_gui.add_button("Load model", lambda: load_model_weights("models/" + model_name_input.get() + ".h5", model, chess_gui), 0, 725)

    chess_gui.start()


