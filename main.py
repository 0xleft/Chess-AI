import os
from core.gui import ChessGUI
from keras.saving.save import load_model, save_model
from core.model import create_model
from core import train
import chess

if __name__ == '__main__':
    if os.path.exists("models/model.h5"):
        model = load_model("models/model.h5")
    else:
        print("Creating new model")
        model = create_model()

    chess_gui = ChessGUI(1000, 1000)
    chess_gui.draw(chess.Board())
    training_options = chess_gui.add_dropdown_menu(["itself", "chesscom"], 100, 0)
    start_training_button = chess_gui.add_button("Start training", lambda: train.set_training_mode(training_options.get(), model, chess_gui), 0, 0)
    stop_training_button = chess_gui.add_button("Stop training", lambda: train.stop_training(), 0, 25)
    save_model_button = chess_gui.add_button("Save model", lambda: save_model(model, "models/model.h5"), 0, 50)
    chess_gui.start()
