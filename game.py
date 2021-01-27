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


def manhattan_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x2-x1) + abs(y2-y1)


def state_distance(state1, state2):
    return (int(state1.dirt.sum())
            - int(state2.dirt.sum())
            + manhattan_distance(state1.pos, state2.pos))


def moves(board_layout, state):
    h, w = board_layout.shape
    x, y = state.pos

    if state.dirt[y, x]:
        new_dirt = state.dirt[:]
        new_dirt[y, x] -= 1
        yield "Clean", State((x, y), new_dirt)

    moves = {
        "Right": ( 1,  0),
        "Down":  ( 0,  1),
        "Left":  (-1,  0),
        "Up":    ( 0, -1)
    }
    for move, (dx, dy) in moves.items():
        new_x, new_y = x+dx, y+dy
        if 0 <= new_x < w and 0 <= new_y < h and board_layout[new_y, new_x]:
            yield move, State((new_x, new_y), state.dirt)


def test():
    board = np.array([ # TODO: get from vision
        [0, 1, 2],
        [3, 4, 5],
        [0, 1, 4]])

    layout, dirt, start_pos, final_pos = parse_board("model/classes", board)
    start_state = State(start_pos, dirt)
    final_state = State(final_pos, dirt*0)

    print(layout)
    print(start_state)
    print(final_state)

    print(list(moves(layout, start_state)))


if __name__ == "__main__":
    test()
