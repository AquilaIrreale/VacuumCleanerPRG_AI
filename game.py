import sys
import math

from time import time_ns
from itertools import count
from heapq import heappush, heappop
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

import vision
from vision import read_board, LetterRecognizerNN


class PQueue:
    @dataclass(order=True)
    class Entry:
        p: Any
        count: int
        v: Any
        deleted: bool = False

    def __init__(self):
        self.heap = []
        self.dict = {}
        self.c = count()

    def __bool__(self):
        return bool(self.dict)

    def remove(self, v):
        entry = self.dict.pop(v)
        entry.deleted = True

    def push(self, p, v):
        if v in self.dict:
            self.remove(v)
        entry = PQueue.Entry(p, next(self.c), v)
        self.dict[v] = entry
        heappush(self.heap, entry)

    def pop(self):
        while True:
            entry = heappop(self.heap)
            if not entry.deleted:
                del self.dict[entry.v]
                return entry.p, entry.v


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


def parse_board(classes_or_labels, board):
    board = np.array(board)

    if isinstance(classes_or_labels, dict):
        classes = dict((v, k) for k, v in classes_or_labels.items())
    else:
        classes = {}
        with open(classes_or_labels) as f:
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


class NoSolution(Exception):
    pass


def uninformed_graph_search(board_layout, start_state, final_state, lifo=False):
    visited = {start_state: (0, None, None)}
    if start_state == final_state:
        return visited
    frontier = deque()
    frontier.append(start_state)
    frontier_set = {start_state}
    n_nodes = 1

    def update():
        print(f"\rNodes expanded: {n_nodes:-10}", end="", flush=True)
    update_period = 500000000
    last_update = time_ns()
    update()

    while frontier:
        state = frontier.pop() if lifo else frontier.popleft()
        frontier_set.discard(state)
        cur_dist, _, _ = visited[state]
        for move, dest_state in moves(board_layout, state):
            if dest_state in visited or dest_state in frontier_set:
                continue
            frontier.append(dest_state)
            frontier_set.add(dest_state)
            n_nodes += 1
            if time_ns() > last_update + update_period:
                last_update += update_period
                update()
            visited[dest_state] = cur_dist+1, move, state
            if dest_state == final_state:
                update()
                print()
                print(f"Solution cost: {cur_dist+1}")
                return visited
    update()
    print()
    raise NoSolution


def informed_graph_search(board_layout, start_state, final_state, heuristic):
    frontier = PQueue()
    frontier.push(0, start_state)
    expanded = defaultdict(lambda: (math.inf, None, None)) # (cost, move, from_state)
    expanded[start_state] = (0, None, None)

    def update():
        print(f"\rNodes expanded: {len(expanded):-10}", end="", flush=True)
    update_period = 500000000
    last_update = time_ns()
    update()

    while frontier:
        _, state = frontier.pop()
        cur_cost, _, _ = expanded[state]
        if state == final_state:
            update()
            print()
            print(f"Solution cost: {cur_cost}")
            return expanded
        for move, dest_state in moves(board_layout, state):
            dest_cost, _, _ = expanded[dest_state]
            if cur_cost + 1 < dest_cost:
                expanded[dest_state] = cur_cost + 1, move, state
                frontier.push(cur_cost+1+heuristic(dest_state, final_state), dest_state)
                if time_ns() > last_update + update_period:
                    last_update += update_period
                    update()
    update()
    print()
    raise NoSolution


def solution_path(nodes, final_state):
    path = []
    state = final_state
    while True:
        _, move, prev_state = nodes[state]
        if move is None:
            break
        path.append((state, move))
        state = prev_state
    path.append((state, None))
    path.reverse()
    return path


def print_path(layout, path):
    try:
        start_state, _ = path[0]
        final_state, _ = path[-1]
    except IndexError:
        return
    for state, move in path:
        if move is not None:
            print()
            print(move)
        print_board(layout, start_state.pos, final_state.pos, state)
    print()


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
    model = LetterRecognizerNN(sys.argv[1])
    print()
    board = read_board(sys.argv[2], model)
    vision.print_board(board, model.labels)
    print(flush=True)

    layout, dirt, start_pos, final_pos = parse_board("model/classes", board)
    start_state = State(start_pos, dirt)
    final_state = State(final_pos, dirt*0)

    print("Uninformed (fifo) graph search")
    try:
        nodes = uninformed_graph_search(layout, start_state, final_state)
    except KeyboardInterrupt:
        print()
    except NoSolution:
        print("No solution found!")
    else:
        path = solution_path(nodes, final_state)
        print_path(layout, path)

    print("Informed (A*) graph search")
    try:
        nodes = informed_graph_search(layout, start_state, final_state, state_distance)
    except KeyboardInterrupt:
        print()
    except NoSolution:
        print("No solution found!")
    else:
        path = solution_path(nodes, final_state)
        print_path(layout, path)


if __name__ == "__main__":
    test()
