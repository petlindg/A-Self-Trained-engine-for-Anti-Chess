from chess import Chessboard, Piece, Color


import random


def generate_random_state(list_pieces: list):
    """Generates a random state based on a list of pieces

    :param list_pieces: List[(Color, Piece)], a list of tuples that are to be included in the state
    :return: String, fen notation string for creating a new chessboard
    """
    full_list = list_pieces + [(None, None)] * (64-len(list_pieces))

    random.shuffle(full_list)
    fen_notation = ''
    empty_counter = 0

    print(full_list)

    for i, (c, p) in enumerate(full_list):
        # if we reach end of a row
        if i != 0 and i % 8 == 0:
            if empty_counter > 0:
                fen_notation = fen_notation + str(empty_counter)
            empty_counter = 0
            fen_notation = fen_notation + '/'

        if p is not None:
            if p == Piece.PAWN:
                if c == Color.WHITE:
                    fen_notation = add_text(fen_notation, 'P', empty_counter)
                else:
                    fen_notation = add_text(fen_notation, 'p', empty_counter)

            elif p == Piece.KNIGHT:
                if c == Color.WHITE:
                    fen_notation = add_text(fen_notation, 'N', empty_counter)
                else:
                    fen_notation = add_text(fen_notation, 'n', empty_counter)

            elif p == Piece.BISHOP:
                if c == Color.WHITE:
                    fen_notation = add_text(fen_notation, 'B', empty_counter)
                else:
                    fen_notation = add_text(fen_notation, 'b', empty_counter)

            elif p == Piece.ROOK:
                if c == Color.WHITE:
                    fen_notation = add_text(fen_notation, 'R', empty_counter)
                else:
                    fen_notation = add_text(fen_notation, 'r', empty_counter)

            elif p == Piece.QUEEN:
                if c == Color.WHITE:
                    fen_notation = add_text(fen_notation, 'Q', empty_counter)
                else:
                    fen_notation = add_text(fen_notation, 'q', empty_counter)

            elif p == Piece.KING:
                if c == Color.WHITE:
                    fen_notation = add_text(fen_notation, 'K', empty_counter)
                else:
                    fen_notation = add_text(fen_notation, 'k', empty_counter)
            empty_counter = 0

        else:
            empty_counter += 1

    color = random.choice(['w', 'b'])
    fen_notation = fen_notation + ' ' + color + ' - 0 1'
    return fen_notation


def add_text(current_txt, value, counter):
    """Helper Function that adds the notation text to the current text

    :param current_txt: String, Current notation that has already been done
    :param value: String, the new character to add to the current text
    :param counter: Int, counter for empty spaces
    :return: String, the new string
    """
    # only include the empty counter if the counter is above 0, else ignore it
    if counter > 0:
        return current_txt + str(counter) + value
    else:
        return current_txt + value


def main():
    list_pieces = [(Color.WHITE, Piece.PAWN), (Color.BLACK, Piece.KING), (Color.WHITE, Piece.ROOK)]
    note = generate_random_state(list_pieces)
    print(note)
    board = Chessboard(note)
    print(board)


if __name__ == '__main__':
    main()