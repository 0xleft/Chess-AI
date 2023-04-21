import chess


def convert_to_int(board):
    l = [None] * 64
    for sq in chess.scan_reversed(board.occupied_co[chess.WHITE]):  # Check if white
        l[sq] = board.piece_type_at(sq)
    for sq in chess.scan_reversed(board.occupied_co[chess.BLACK]):  # Check if black
        l[sq] = -board.piece_type_at(sq)
    return [0 if v is None else v for v in l]
