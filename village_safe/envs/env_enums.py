from enum import Enum

class ObjectiveState(Enum):
    NOT_REACHED = 0
    REACHED_THIS_FRAME = 1
    REACHED_PAST = 2

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class Zone(Enum):
    C = (10, 10)
    N = (10, 0)
    NE = (20, 0)
    E = (20, 10)
    SE = (20, 20)
    S = (10, 20)
    SW = (0, 20)
    W = (0, 10)
    NW = (0, 0)

class Terrain(Enum):
    VILLAGE = 0
    FOREST = 1
    OCEAN = 2
    MOUNTAIN = 3
    GRASSLAND = 4
    BEACH = 5

class GameObject(Enum):
    EMPTY = 0
    SWORD = 1
    MAGIC_WAND = 2
    FIRE = 3
    TREE = 4
    MERMAID = 5
    DRAGON = 6
    ROCK = 7

class ObservationMap(Enum):
    NOTHING = 0
    SWORD = 1
    FIRE = 2
    ROCK = 3
    WATER = 4

