from .env_enums import Zone, Terrain, GameObject
import numpy as np
from .Tile import Tile

class WorldBuilder:
    def __init__(self, size, seed):
        self.size = size
        self.seed = seed
        self.map = [[Tile() for _ in range(size)] for _ in range(size)]
        self.zones = {zone.name: Terrain.GRASSLAND for zone in Zone}
        
        self.forest_zone = None
        self.mountain_zone = None
        self.ocean_zones = []
        self.village_zone = None

        self.mermaid_coords = None
        self.sword_coords = None
        self.dragon_coords = None
        self.fire_coords = []
        
        self.generate_zones()
        self.generate_world()
        self.place_mermaid()
        self.place_sword()
        self.place_dragon()

    def generate_world(self):
        for zone in Zone:
            zone_offset = Zone[zone.name].value
            zone_terrain = self.zones[zone.name]
            if zone_terrain == Terrain.VILLAGE:
                self._populate_zone_tiles(zone_offset, Terrain.VILLAGE)
            elif zone_terrain == Terrain.OCEAN:
                self._populate_zone_tiles(zone_offset, Terrain.OCEAN)
            elif zone_terrain == Terrain.FOREST:
                self._populate_zone_tiles(zone_offset, Terrain.FOREST)
            elif zone_terrain == Terrain.MOUNTAIN:
                self._populate_zone_tiles(zone_offset, Terrain.MOUNTAIN)
            else:
                self._populate_zone_tiles(zone_offset, Terrain.GRASSLAND)

    def return_object_coords(self):
        return self.mermaid_coords, self.sword_coords, self.dragon_coords, self.fire_coords

    def generate_zones(self):
        # Generate Village Zone
        self.zones[Zone.C.name] = Terrain.VILLAGE
        self.village_zone = Zone.C

        # Generate Ocean Zone
        NSEW_zones = [Zone.N.name, Zone.E.name, Zone.S.name, Zone.W.name] # North, South, East, West zones, so that ocean is not in the middle
        ocean_zone_name = self.get_random_valid_zone([Zone.C, *NSEW_zones])
        is_col = np.random.randint(2)
        if is_col:
            if "E" in ocean_zone_name:
                self.zones[Zone.NE.name] = Terrain.OCEAN
                self.zones[Zone.E.name] = Terrain.OCEAN
                self.zones[Zone.SE.name] = Terrain.OCEAN
                self.ocean_zones = [Zone.NE.name, Zone.E.name, Zone.SE.name]
            else:
                self.zones[Zone.NW.name] = Terrain.OCEAN
                self.zones[Zone.W.name] = Terrain.OCEAN
                self.zones[Zone.SW.name] = Terrain.OCEAN
                self.ocean_zones = [Zone.NW.name, Zone.W.name, Zone.SW.name]
        else:
            if "N" in ocean_zone_name:
                self.zones[Zone.NW.name] = Terrain.OCEAN
                self.zones[Zone.N.name] = Terrain.OCEAN
                self.zones[Zone.NE.name] = Terrain.OCEAN
                self.ocean_zones = [Zone.NW.name, Zone.N.name, Zone.NE.name]
            else:
                self.zones[Zone.SW.name] = Terrain.OCEAN
                self.zones[Zone.S.name] = Terrain.OCEAN
                self.zones[Zone.SW.name] = Terrain.OCEAN
                self.ocean_zones = [Zone.SW.name, Zone.S.name, Zone.SE.name]

        # Generate Forest Zone
        forest_zone_name = self.get_random_valid_zone()
        self.zones[forest_zone_name] = Terrain.FOREST
        self.forest_zone = Zone[forest_zone_name]
        
        # Generate Mountain Zone
        mountain_zone_name = self.get_random_valid_zone()
        self.zones[mountain_zone_name] = Terrain.MOUNTAIN
        self.mountain_zone = Zone[mountain_zone_name]

    def place_sword(self):
        buffer = 2
        forest_offset = self.forest_zone.value
        coord_in_forest = (np.random.randint(buffer, self.size//3 - buffer), np.random.randint(buffer, self.size//3 - buffer))

        map_coordinates = (forest_offset[0] + coord_in_forest[0], forest_offset[1] + coord_in_forest[1])
        self.sword_coords = map_coordinates
        self.map[map_coordinates[0]][map_coordinates[1]].object = GameObject.SWORD

        # surround sword with fire
        for row in range(map_coordinates[0] - 1, map_coordinates[0] + 2):
            for col in range(map_coordinates[1] - 1, map_coordinates[1] + 2):
                if self.map[row][col].object == GameObject.EMPTY:
                    self.map[row][col].object = GameObject.FIRE
                    self.fire_coords.append((row, col))

    
    def place_dragon(self):
        buffer = 2
        mountain_offset = self.mountain_zone.value
        coord_in_mountain = (np.random.randint(buffer, self.size//3 - buffer), np.random.randint(buffer, self.size//3 - buffer))

        map_coordinates = (mountain_offset[0] + coord_in_mountain[0], mountain_offset[1] + coord_in_mountain[1])
        self.dragon_coords = map_coordinates
        self.map[map_coordinates[0]][map_coordinates[1]].object = GameObject.DRAGON



    def place_mermaid(self):
        buffer = 2
        mountain_offset = self.mountain_zone.value
        coord_in_mountain = (np.random.randint(buffer, self.size//3 - buffer), np.random.randint(buffer, self.size//3 - buffer))

        map_coordinates = (mountain_offset[0] + coord_in_mountain[0], mountain_offset[1] + coord_in_mountain[1])
        self.mermaid_coords = map_coordinates
        self.map[map_coordinates[0]][map_coordinates[1]].object = GameObject.MERMAID

    def get_random_valid_zone(self, invalid_zones=None):
        if invalid_zones is None:
            valid_zones = [zone.name for zone in Zone if self.zones[zone.name] == Terrain.GRASSLAND]
        else:
            valid_zones = [zone.name for zone in Zone if self.zones[zone.name] not in invalid_zones]
        print(valid_zones)
        return valid_zones[np.random.choice(len(valid_zones))]
    
    def get_random_zone_tile(self, zone_coords):
        col_range = range(zone_coords[0] * 10, zone_coords[0] * 10 + 10)
        row_range = range(zone_coords[1] * 10, zone_coords[1] * 10 + 10)
        valid_tiles = [(x, y) for x in row_range for y in col_range if map[x][y] == Tiles.EMPTY]
        return valid_tiles[np.random.randint(len(valid_tiles))]

    def _populate_zone_tiles(self, zone_offset, terrain):
        for row in range(zone_offset[0], zone_offset[0] + 10):
            for col in range(zone_offset[1], zone_offset[1] + 10):
                self.map[row][col].terrain = terrain