from collections import namedtuple

import numpy as np


State = namedtuple("State", ("pos", "dirt"))


def parse_board(classes_file, board):
    board = np.array(board)

    classes = {}
    with open(classes_file) as f:
        for line in f:
            num, label = line.split()
            classes[label] = int(num)

    not_wall = np.vectorize(lambda x: x != classes["X"])
    board_layout = not_wall(board)

    start_pos = None
    final_pos = None
    dirt = np.zeros(board.shape, dtype=np.uint8)
    dirt_levels = {
        classes["D"]: 1,
        classes["V"]: 2
    }
    for y, line in enumerate(board):
        for x, cell_value in enumerate(line):
            dirt[y, x] = dirt_levels.get(cell_value, 0)
            if cell_value == classes["S"]:
                if start_pos is not None:
                    raise ValueError("Multiple starting positions in board")
                start_pos = (x, y)
            if cell_value == classes["F"]:
                if final_pos is not None:
                    raise ValueError("Multiple final positions in board")
                final_pos = (x, y)

    if start_pos is None:
        raise ValueError("Missing starting position in board")
    if final_pos is None:
        raise ValueError("Missing final position in board")

    return board_layout, dirt, start_pos, final_pos


def solve():
    board = np.array([ # TODO: get from vision
        [0, 1, 2],
        [3, 4, 5],
        [0, 1, 4]])

    board, dirt, start_pos, final_pos = parse_board("model/classes", board)
    start_state = State(start_pos, dirt)
    final_state = State(final_pos, dirt*0)

    print(board)
    print(start_state)
    print(final_state)


if __name__ == "__main__":
    solve()
