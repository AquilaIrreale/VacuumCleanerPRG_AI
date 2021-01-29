from collections import deque
import numpy as np

import vision
from vision import read_board, LetterRecognizerNN


class State:
    def __init__(self, pos, dirt):
        self.pos = pos
        self.dirt = np.copy(dirt)
        self.dirt.flags.writeable = False

    def __eq__(self, other):
        return self.pos == other.pos and np.array_equal(self.dirt, other.dirt)

    def __hash__(self):
        return hash((self.pos, self.dirt.shape, bytes(self.dirt.data)))

    def __repr__(self):
        return f"State({self.pos!r}, {self.dirt!r})"


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
        new_dirt = np.copy(state.dirt)
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


def uninformed_graph_search(board_layout, start_state, final_state, lifo=False):
    if start_state == final_state:
        return None, 0

    visited = {}
    frontier = deque()
    frontier.append(start_state)
    expanded = 1

    while frontier:
        state = frontier.pop() if lifo else frontier.popleft()
        for move, dest_state in moves(board_layout, state):
            if dest_state in visited or dest_state in frontier:
                continue
            frontier.append(dest_state)
            expanded += 1
            visited[dest_state] = move, state
            if dest_state == final_state:
                return visited, expanded

    return None, expanded


def print_board(layout, start_pos, final_pos, state):
    h, w = layout.shape

    print(f"+{'-'*(w*2)}+")
    for y in range(h):
        print('|', end="")
        for x in range(w):
            if not layout[y, x]:
                print("[]", end="")
                continue

            if (x, y) == start_pos:
                cell = "S"
            elif (x, y) == final_pos:
                cell = "F"
            elif state.dirt[y, x] == 2:
                cell = "V"
            elif state.dirt[y, x] == 1:
                cell = "D"
            else:
                cell = " "

            if state.pos == (x, y):
                print(f"{cell}*", end="")
            else:
                print(cell*2, end="")
        print('|')
    print(f"+{'-'*(w*2)}+")


def test():
    model = LetterRecognizerNN("model")
    print()
    board = read_board("images/digital.png", model)
    vision.print_board(board, model.labels)
    print(flush=True)

    layout, dirt, start_pos, final_pos = parse_board("model/classes", board)
    start_state = State(start_pos, dirt)
    final_state = State(final_pos, dirt*0)

    visited, expanded = uninformed_graph_search(layout, start_state, final_state)

    if visited is None:
        print("No solution found!")
        return

    path = []
    state = final_state
    while state != start_state:
        move, prev_state = visited[state]
        path.append((state, move))
        state = prev_state
    path.append((state, None))
    path.reverse()

    for state, move in path:
        if move is not None:
            print()
            print(move)
        print_board(layout, start_pos, final_pos, state)

    print()
    print(f"Nodes expanded: {expanded}")


if __name__ == "__main__":
    test()
