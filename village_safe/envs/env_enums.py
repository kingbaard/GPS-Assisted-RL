from enum import Enum

class ObjectiveStatus(Enum):
    NOT_REACHED = 0
    REACHED_THIS_FRAME = 1
    REACHED_PAST = 2

class GoalState(Enum):
    NO_DRAGON = 0
    HAS_SWORD = 1
    NO_FIRE = 2
    IN_BOAT = 3

# named after the starting objective
class StartState(Enum):
    BOAT = 0
    MERMAID = 1
    SWORD = 2
    DRAGON = 3

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

# For 15x15 map
class Zone(Enum):
    C = (5, 5)
    N = (5, 0)
    NE = (10, 0)
    E = (10, 5)
    SE = (10, 10)
    S = (5, 10)
    SW = (0, 10)
    W = (0, 5)
    NW = (0, 0)

# For 9x9 map
# class Zone(Enum):
#     C = (3, 3)
#     N = (3, 0)
#     NE = (6, 0)
#     E = (6, 3)
#     SE = (6, 6)
#     S = (3, 6)
#     SW = (0, 6)
#     W = (0, 3)
#     NW = (0, 0)

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