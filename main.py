#!/usr/bin/env python3

import sys
import time
from importlib import resources
from threading import Thread, Lock

import pygame
from pygame import display, transform, event, Color, Rect

import assets
from vision import LetterRecognizerNN


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
    def __init__(self, frequency):
        self.target = None
        self.period = 10**9//frequency

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
        self.tile_size = self.BASE_TILE_SIZE * scale

        tileset = self.image_from_resource(assets, "tileset.png").convert()
        vacuum  = self.image_from_resource(assets, "vacuum.png").convert()
        dirt    = self.image_from_resource(assets, "dirt.png").convert()

        if scale > 1:
            tileset = self.scale(tileset, scale)
            vacuum  = self.scale(vacuum, scale)
            dirt    = self.scale(dirt, scale)

        for surface in tileset, vacuum, dirt:
            surface.set_colorkey(Color(255, 0, 255))

        self.default_tile = self.get_tile(tileset, 0, 0)
        self.start_tile   = self.get_tile(tileset, 1, 0)
        self.finish_tile  = self.get_tile(tileset, 2, 0)
        self.wall_tile_l  = self.get_tile(tileset, 0, 1)
        self.wall_tile_mr = self.get_tile(tileset, 1, 1)

        self.vacuum_a = [self.get_tile(vacuum, i, 0) for i in range(2)]
        self.vacuum_b = [self.get_tile(vacuum, i, 1) for i in range(2)]

        self.dirt = [self.get_tile(dirt, i, 0) for i in range(2)]


class GameQuit(Exception):
    pass


class BaseModule:
    def __init__(self, clock_freq=30):
        self.clock = Clock(clock_freq)

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
        screen = display.set_mode(size)
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
        self.thread = Thread(target=self.worker, daemon=True)

    def worker(self):
        self.model = LetterRecognizerNN(self.model_path)

    def start(self):
        super().start()
        self.thread.start()

    def update(self):
        if not self.thread.is_alive():
            return self.model


def main(model_path, board_image, algorithm):
    display.init()
    model = ModelLoadingModule(model_path).run()
    print(model)


if __name__ == "__main__":
    _, model, board_image, *rest = sys.argv
    if len(rest) == 0:
        algorithm = "A*"
    else:
        algorithm, = rest
    main(model, board_image, algorithm)
