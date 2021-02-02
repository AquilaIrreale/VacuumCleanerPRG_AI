import os
import sys
import time
import random

from functools import partial
from itertools import product
from importlib import resources
from threading import Thread, Lock

import pygame
from pygame import display, transform, event, Color, Rect, Surface

import game
import vision
import assets

from game import State
from vision import LetterRecognizerNN


def get_display_size():
    mode_info = display.Info()
    return mode_info.current_w, mode_info.current_h


set_mode_first_time = True

def set_mode_if_needed(size):
    global set_mode_first_time
    if not set_mode_first_time:
        cur_size = get_display_size()
        if cur_size == size:
            return display.get_surface()

    set_mode_first_time = False
    display.quit()
    display.init()
    w, h = size
    dw, dh = get_display_size()
    x = (dw-w)//2
    y = (dh-h)//2
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
    return display.set_mode(size)


class Timer:
    def __init__(self):
        self.target = None
        self.period = None

    def set(self, period_ms):
        self.period = period_ms * 10**6
        self.target = time.time_ns() + self.period

    def tick(self):
        if time.time_ns() < self.target:
            return False
        self.target += self.period
        return True


class Clock:
    def __init__(self, freq=None, period=None):
        self.target = 0
        if freq is not None:
            self.period = 10**9 // freq
        elif period is not None:
            self.period = period * 10**6
        else:
            raise TypeError("Either freq or period must be specified")

    def start(self):
        self.target = time.time_ns() + self.period

    def time_remaining(self):
        t = (self.target - time.time_ns()) // 10**6
        return t if t >= 0 else 0

    def advance(self):
        self.target += self.period


class Assets:
    BASE_TILE_SIZE = 16

    @staticmethod
    def image_from_resource(module, res_name):
        with resources.open_binary(module, res_name) as f:
            return pygame.image.load(f)

    @staticmethod
    def scale(surface, scale):
        w, h = surface.get_size()
        return transform.scale(surface, (w*scale, h*scale))

    def get_tile(self, surface, x, y):
        return surface.subsurface(Rect(
                    x * self.tile_size,
                    y * self.tile_size,
                    self.tile_size,
                    self.tile_size))

    def __init__(self, scale=1):
        self.pixel_size = scale
        self.tile_size = self.BASE_TILE_SIZE * scale

        tileset = self.image_from_resource(assets, "tileset.png").convert()
        vacuum  = self.image_from_resource(assets, "vacuum.png").convert()
        dirt    = self.image_from_resource(assets, "dirt.png").convert()

        if scale > 1:
            tileset = self.scale(tileset, scale)
            vacuum  = self.scale(vacuum, scale)
            dirt    = self.scale(dirt, scale)

        for surface in tileset, vacuum:
            surface.set_colorkey(Color(255, 0, 255))

        self.default_tile = self.get_tile(tileset, 0, 0)
        self.start_tile   = self.get_tile(tileset, 1, 0)
        self.finish_tile  = self.get_tile(tileset, 2, 0)
        self.wall_tile_l  = self.get_tile(tileset, 0, 1)
        self.wall_tile_mr = self.get_tile(tileset, 1, 1)

        self.vacuum = [
                [self.get_tile(vacuum, j, i) for j in range(2)]
                for i in range(2)]

        self.dirt = [self.get_tile(dirt, i, 0) for i in range(3)]

    def grid_pos(self, pos):
        x, y = pos
        return self.tile_size * x, self.tile_size * y


class GameQuit(Exception):
    pass


class BaseModule:
    def __init__(self, clock_freq=30):
        self.clock = Clock(freq=clock_freq)

    def run(self):
        self.start()
        self.clock.start()
        while True:
            while True:
                wait_time = self.clock.time_remaining()
                if wait_time > 0:
                    e = event.wait(wait_time)
                else:
                    e = event.poll()
                if e.type == pygame.NOEVENT:
                    break
                self.event(e)
            self.clock.advance()

            update_ret = self.update()
            if update_ret is not None:
                return update_ret

            self.render()

    def start(self):
        raise NotImplementedError

    def event(self, e):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError


class SplashScreenModule(BaseModule):
    def __init__(self, splash_image_asset):
        super().__init__(5)
        self.splash_image_asset = splash_image_asset

    def start(self):
        with resources.open_binary(assets, self.splash_image_asset) as f:
            splash_surface = pygame.image.load(f)
        size = splash_surface.get_size()
        screen = set_mode_if_needed(size)
        screen.blit(splash_surface, (0, 0))
        display.flip()

    def event(self, e):
        if e.type == pygame.QUIT:
            raise GameQuit

    def render(self):
        pass


class ModelLoadingModule(SplashScreenModule):
    def __init__(self, model_path):
        super().__init__("load_splash.png")
        self.model_path = model_path
        self.model = None
        self.thread = Thread(target=self.worker)

    def worker(self):
        self.model = LetterRecognizerNN(self.model_path)

    def start(self):
        super().start()
        self.thread.start()

    def update(self):
        if not self.thread.is_alive():
            return self.model


class ReadingBoardModule(SplashScreenModule):
    def __init__(self, model, board_image_path):
        super().__init__("reading_board_splash.png")
        self.model = model
        self.board_image_path = board_image_path
        self.board_data = None
        self.thread = Thread(target=self.worker, daemon=True)

    def worker(self):
        board = vision.read_board(self.board_image_path, self.model)
        self.board_data = game.parse_board(self.model.labels, board)

    def start(self):
        super().start()
        self.thread.start()

    def update(self):
        if not self.thread.is_alive():
            return self.board_data


class SolvingModule(SplashScreenModule):
    def __init__(self, board_layout, start_dirt, start_pos, final_pos, algorithm):
        super().__init__("solving_splash.png")
        self.board_layout = board_layout
        self.start_dirt = start_dirt
        self.start_pos = start_pos
        self.final_pos = final_pos
        self.algorithm = algorithm
        self.path = None
        self.thread = Thread(target=self.worker, daemon=True)

    algorithms = {
        "bfs": partial(game.uninformed_graph_search, lifo=True),
        "dfs": partial(game.uninformed_graph_search, lifo=False),
        "a*":  partial(game.informed_graph_search, heuristic=game.state_distance)
    }

    def worker(self):
        solve = self.algorithms[self.algorithm.casefold()]
        start_state = State(self.start_pos, self.start_dirt)
        final_state = State(self.final_pos, self.start_dirt*0)
        nodes = solve(self.board_layout, start_state, final_state)
        self.path = game.solution_path(nodes, final_state)

    def start(self):
        super().start()
        self.thread.start()

    def update(self):
        if not self.thread.is_alive():
            return self.path


class Dirt:
    def __init__(self, assets, pos, level):
        self.assets = assets
        self.pos = pos
        self._level = level
        self.sprite = assets.dirt[level].copy()
        self.sprite.set_colorkey(Color(255, 0, 255))
        self.animation_timer = Timer()
        self.animation_coords = [(x, y) for x, y in product(range(16), repeat=2)]
        self.animation_index = len(self.animation_coords)
        random.shuffle(self.animation_coords)

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self._level = value
        self.animation_index = 0
        self.animation_timer.set(1000//len(self.animation_coords))

    def update(self):
        while (self.animation_index < len(self.animation_coords)
                and self.animation_timer.tick()):
            x, y = self.animation_coords[self.animation_index]
            pixel = Rect(
                    x * self.assets.pixel_size,
                    y * self.assets.pixel_size,
                    self.assets.pixel_size,
                    self.assets.pixel_size)
            self.sprite.blit(self.assets.dirt[self._level], pixel, pixel)
            self.animation_index += 1

    def render(self, dest_surf):
        dest_surf.blit(self.sprite, self.assets.grid_pos(self.pos))


class Vacuum:
    def __init__(self, assets, pos):
        self.assets = assets
        self._pos = pos
        self.old_pos = pos
        self.frame_timer = Timer()
        self.frame_timer.set(500)
        self.move_clock = Clock(period=1000)
        self.animation_state = 0
        self.animation_frame = 0

    @staticmethod
    def animation_function(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1
        # Ruffini-Horner of 6x^5 - 15x^4 + 10x^3
        return ((6 * x - 15) * x + 10) * x**3

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self.old_pos = self._pos
        self._pos = value
        self.move_clock.start()

    def update(self):
        if self.frame_timer.tick():
            frames = self.assets.vacuum[self.animation_state]
            self.animation_frame = (self.animation_frame+1) % len(frames)

    def render(self, dest_surf):
        x, y = self.assets.grid_pos(self._pos)
        t = self.move_clock.time_remaining()

        if t > 0:
            old_x, old_y = self.assets.grid_pos(self.old_pos)
            dx, dy = x-old_x, y-old_y
            c = self.animation_function(1-t/1000)
            x = int(old_x + c*dx)
            y = int(old_y + c*dy)

        frames = self.assets.vacuum[self.animation_state]
        dest_surf.blit(frames[self.animation_frame], (x, y))


class MainGameModule(BaseModule):
    def __init__(self, desktop_size, board_layout, path):
        super().__init__(clock_freq=60)
        self.desktop_size = desktop_size
        self.board_layout = board_layout
        self.path = path
        self.path_index = 0
        self.state_advance_timer = Timer()
        self.screen = None
        self.background = None
        self.dirt = {}
        self.vacuum = None

    SIZE_FACT = .9 # Max window size relative to desktop

    def start(self):
        n, m = self.board_layout.shape
        dw, dh = self.desktop_size
        scale_factor = min(
            int(dh * self.SIZE_FACT) // n // Assets.BASE_TILE_SIZE,
            int(dw * self.SIZE_FACT) // m // Assets.BASE_TILE_SIZE)
        self.assets = Assets(scale_factor)
        h = self.assets.tile_size * n + scale_factor
        w = self.assets.tile_size * m + scale_factor
        self.screen = set_mode_if_needed((w, h))
        self.background = Surface((w, h))
        self.background.fill(Color(0, 0, 0))

        start_state, _ = self.path[0]
        final_state, _ = self.path[-1]

        for y, row in enumerate(self.board_layout):
            for x, cell in enumerate(row):
                if (x, y) == start_state.pos:
                    tile = self.assets.start_tile
                elif (x, y) == final_state.pos:
                    tile = self.assets.finish_tile
                elif cell:
                    tile = self.assets.default_tile
                elif x > 0 and not self.board_layout[y, x-1]:
                    tile = self.assets.wall_tile_mr
                else:
                    tile = self.assets.wall_tile_l
                pos_x = x * self.assets.tile_size
                pos_y = y * self.assets.tile_size
                self.background.blit(tile, (pos_x, pos_y))

                dirt_level = start_state.dirt[y, x]
                if dirt_level:
                    self.dirt[(x, y)] = Dirt(self.assets, (x, y), dirt_level)

        self.vacuum = Vacuum(self.assets, start_state.pos)
        self.state_advance_timer.set(1500)

    def event(self, e):
        if e.type == pygame.QUIT:
            raise GameQuit

    def update(self):
        if self.state_advance_timer.tick():
            try:
                old_state, _ = self.path[self.path_index]
                self.path_index += 1
                new_state, move = self.path[self.path_index]
            except IndexError as e:
                return

            if new_state.pos != old_state.pos:
                self.vacuum.pos = new_state.pos

            for y, row in enumerate(new_state.dirt):
                for x, d in enumerate(row):
                    if d < old_state.dirt[y, x]:
                        self.dirt[(x, y)].level = d

        for d in self.dirt.values():
            d.update()
        self.vacuum.update()

    def render(self):
        self.screen.blit(self.background, (0, 0))
        for d in self.dirt.values():
            d.render(self.screen)
        self.vacuum.render(self.screen)
        display.flip()


def main(model_path, board_image_path, algorithm):
    display.init()
    desktop_size = get_display_size()
    try:
        model = ModelLoadingModule(model_path).run()
        print()
        board_layout, start_dirt, start_pos, final_pos = ReadingBoardModule(model, board_image_path).run()
        path = SolvingModule(board_layout, start_dirt, start_pos, final_pos, algorithm).run()
        print()
        MainGameModule(desktop_size, board_layout, path).run()
    except (GameQuit, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    _, model, board_image_path, *rest = sys.argv
    if len(rest) == 0:
        algorithm = "A*"
    else:
        algorithm, = rest
    main(model, board_image_path, algorithm)
