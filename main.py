import time
from importlib import resources

import pygame
from pygame import display, Color, Rect

import assets


class Assets:
    def __init__(self):
        with resources.open_binary(assets, "tileset.png") as f:
            tileset = pygame.image.load(f).convert()
        with resources.open_binary(assets, "vacuum.png") as f:
            vacuum = pygame.image.load(f).convert()
        with resources.open_binary(assets, "dirt.png") as f:
            dirt = pygame.image.load(f).convert()

        for surface in tileset, vacuum, dirt:
            surface.set_colorkey(Color(255, 0, 255))

        self.default_tile = tileset.subsurface(Rect( 0,  0, 16, 16))
        self.start_tile   = tileset.subsurface(Rect(16,  0, 16, 16))
        self.finish_tile  = tileset.subsurface(Rect(32,  0, 16, 16))
        self.wall_tile_l  = tileset.subsurface(Rect( 0, 16, 16, 16))
        self.wall_tile_mr = tileset.subsurface(Rect(16, 16, 16, 16))

        self.vacuum_default  = [vacuum.subsurface(Rect(i*16,  0, 16, 16)) for i in range(2)]
        self.vacuum_cleaning = [vacuum.subsurface(Rect(i*16, 16, 16, 16)) for i in range(2)]

        self.dirt = [dirt.subsurface(Rect(i*16, 0, 16, 16)) for i in range(2)]


# ASSETS TEST
pygame.init()
screen = display.set_mode((16, 16), pygame.SCALED)
assets = Assets()
for surfaces in vars(assets).values():
    if not isinstance(surfaces, list):
        surfaces = [surfaces]
    for surface in surfaces:
        screen.fill(Color(255, 255, 255))
        screen.blit(surface, (0, 0))
        display.update()
        time.sleep(.5)
