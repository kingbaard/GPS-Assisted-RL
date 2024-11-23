
from dataclasses import dataclass
from .env_enums import Terrain, GameObject

@dataclass
class Tile:
    terrain: Terrain = Terrain.GRASSLAND
    object: GameObject = GameObject.EMPTY

