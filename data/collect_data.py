from core import chess_com_data

full_data = []
for archive in chess_com_data.get_player_archives(chess_com_data.get_players()[10]):
    full_data.append(chess_com_data.collect_player_data(archive))

print(f"Finished collecting {len(full_data)} games")
# save data in files
for data in full_data:
    with open(f"inputs", "a+") as file:
        for game in data:
            file.write(str(game[0].tolist()[0]) + "\n")
    with open(f"outputs", "a+") as file:
        for game in data:
            file.write(str(game[1].tolist()[0]) + "\n")
