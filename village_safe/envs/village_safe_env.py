from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class Zones(Enum):
    VILLAGE = 0
    FOREST = 1
    OCEAN = 2
    MOUNTAIN = 3
    GRASSLAND = 4

# Tile-types that the agent must be aware of
class Tiles(Enum):
    EMPTY = 0
    OCEAN = 1
    TREE_FIRE = 2
    TREE_SAFE = 3
    DRAGON = 4
    SWORD = 5


class VillageSafeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.size = 30  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.map = self._generate_world()
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "sword": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "has-sword": spaces.Discrete(2), # 0: not in inventory, 1: in inventory
                "dragon": spaces.Box(0, self.size - 1, shape=(4,2), dtype=int),
                # TODO: add other entities
            }
        )

        self.action_space = spaces.Discrete(4) # "right", "up", "left", "down", "right"

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location}

    def _get_info(self):
        return {
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )
        self.map = self._generate_world()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self.map.sword_location) #temporarily using sword location as target
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        

    def _generate_world(self):
        def get_random_valid_zone(valid_zones=None):
            if valid_zones is None:
                valid_zones = [(x, y) for x in range(3) for y in range(3) if zones[x][y] == Zones.GRASSLAND]
            else:
                valid_zones = [(x, y) for x in range(3) for y in range(3) if zones[x][y] in valid_zones]
            return valid_zones[np.random.choice(len(valid_zones))]
        
        def get_random_zone_tile(zone):
            valid_tiles = [(x, y) for x in range(3) for y in range(3) if zones[x][y] == zone]
            return valid_tiles[np.random.randint(len(valid_tiles))]
        
        map = np.zeros((self.size, self.size))

        # Generate high-level zones
        zones = [[Zones.GRASSLAND for _ in range(3)] for _ in range(3)]
        zones[1][1] = Zones.VILLAGE.value # village will always be in center

        ### Generate ocean
        # Reserve a zone for ocean
        NSEW_zones = [zones[1][0], zones[1][2], zones[0][1], zones[2][1]] # North, South, East, West zones, so that ocean is not in the middle
        ocean_coords = get_random_valid_zone([Zones.VILLAGE, *NSEW_zones])
        is_col = np.random.randint(2)
        if is_col:
            for i in range(3):
                zones[i][ocean_coords[1]] = Zones.OCEAN.value
        else:
            for i in range(3):
                zones[ocean_coords[0]][i] = Zones.OCEAN.value

        # Place ocean tiles on map
        # TODO: Make some of the tiles beach tiles
        for i in range(self.size):
            for j in range(self.size):
                if zones[i//10][j//10] == Zones.OCEAN:
                    map[i][j] = Tiles.OCEAN.value

        ### Generate forest
        # Reserve a zone for forest
        forest_coords = get_random_valid_zone([Zones.VILLAGE])
        zones[forest_coords[0]][forest_coords[1]] = Zones.FOREST.value

        # Place sword somewhere in the forest
        sword_coords = get_random_zone_tile(Zones.FOREST)
        map[sword_coords[0]*10:sword_coords[0]*10+10, sword_coords[1]*10:sword_coords[1]*10+10] = Tiles.SWORD

        # Place fire trees in the forest, blocking the sword
        # for i in range(3):
        #     for j in range(3):
        #         if zones[i][j] == Zones.FOREST:
        #             if (i, j) != forest_coords:
        #                 map[i*10:i*10+10, j*10:j*10+10] = Tiles.TREE_FIRE

        ### Generate mountain
        # Reserve a zone for mountain
        mountain_coords = get_random_valid_zone([Zones.VILLAGE])
        zones[mountain_coords[0]][mountain_coords[1]] = Zones.MOUNTAIN.value

        # Place Dragon (takes 4 tiles)
        dragon_coords = get_random_zone_tile(Zones.MOUNTAIN)
        map[dragon_coords[0]*10:dragon_coords[0]*10+10, dragon_coords[1]*10:dragon_coords[1]*10+10] = Tiles.DRAGON.value

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw the map
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i][j] == Tiles.OCEAN:
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 255),
                        pygame.Rect(
                            (i * pix_square_size, j * pix_square_size),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.map[i][j] == Tiles.TREE_FIRE:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            (i * pix_square_size, j * pix_square_size),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.map[i][j] == Tiles.TREE_SAFE:
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        pygame.Rect(
                            (i * pix_square_size, j * pix_square_size),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.map[i][j] == Tiles.DRAGON:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 255),
                        pygame.Rect(
                            (i * pix_square_size, j * pix_square_size),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.map[i][j] == Tiles.SWORD:
                    pygame.draw.rect(
                        canvas,
                        (255, 255, 0),
                        pygame.Rect(
                            (i * pix_square_size, j * pix_square_size),
                            (pix_square_size, pix_square_size),
                        )
                    )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
