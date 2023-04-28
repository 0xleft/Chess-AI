from core import chess_com_data

def get_data(index=0):
    # it goes as follows:
    # [ 1 , 80 ] array of game moves
    # [ 2, 64 ] arrays of next move
    data = []
    # get game from chess com
    player = chess_com_data.get_players()[index]
    games = []
    for archive in chess_com_data.get_player_archives(player):
        games = chess_com_data.collect_player_data_moves(archive)
    return games