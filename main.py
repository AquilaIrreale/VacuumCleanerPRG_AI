import time
from importlib import resources

import pygame
from pygame import display, transform, Color, Rect

import assets


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


# ASSETS TEST
pygame.init()
screen = display.set_mode((16, 16), pygame.SCALED)
assets = Assets()
for surfaces in vars(assets).values():
    if not isinstance(surfaces, list):
        surfaces = [surfaces]
    for surface in surfaces:
        if isinstance(surface, pygame.Surface):
            screen.fill(Color(255, 255, 255))
            screen.blit(surface, (0, 0))
            display.update()
            time.sleep(.5)
