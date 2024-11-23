from enum import Enum

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class Zone(Enum):
    C = (10, 10)
    N = (0, 10)
    NE = (0, 20)
    E = (10, 20)
    SE = (20, 20)
    S = (20, 10)
    SW = (20, 0)
    W = (10, 0)
    NW = (0, 0)

class Terrain(Enum):
    VILLAGE = 0
    FOREST = 1
    OCEAN = 2
    MOUNTAIN = 3
    GRASSLAND = 4

class GameObject(Enum):
    EMPTY = 0
    SWORD = 1
    MAGIC_WAND = 2
    FIRE = 3
    TREE = 4
    MERMAID = 5
    DRAGON = 6
