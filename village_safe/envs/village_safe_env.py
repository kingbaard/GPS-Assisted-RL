import sys
import os

from .env_enums import Actions, Zone, Terrain, GameObject
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from .WorldBuilder import WorldBuilder

class VillageSafeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
        "debug": False,
        "goal_states": {"no-dragon": True, "no-fire": False, "has-sword": False},} # Will be used to train indiviual actions for GPS-assisted agent

    def __init__(self, render_mode=None, control_mode=None):
        self.size = 30  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.world_builder = WorldBuilder(self.size, 0)
        self.world_builder.generate_world()
        self.map = self.world_builder.map
        self.mermaid_location, self.sword_location, self.dragon_location, self.fire_coords = self.world_builder.return_object_coords()
        self.has_sword = 0
        self.agent_dead = False
        self.goal_states = self.metadata["goal_states"]
        self.current_objective = None # Will be determined by GPS


        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "sword": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "fire": spaces.Box(low=0, high=self.size - 1, shape=(15, 2), dtype=int),
                "has-sword": spaces.Discrete(2), # 0: not in inventory, 1: in inventory
                "dragon": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
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

        if render_mode == "human":
            # load sprites
            self._load_sprites()

        assert control_mode is None or control_mode in ["human", "agent"]
        self.control_mode = control_mode

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
        return {
            "agent": self._agent_location,
            "fire": self.fire_coords,
            "sword": self.sword_location,
            "has-sword": self.has_sword,
            "dragon": self.dragon_location,
                }

    def _get_info(self):
        return {
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(10, 20, size=2, dtype=int)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )
        self.world_builder.generate_world()
        self.map = self.world_builder.map
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        did_win = False
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        prospective_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if action will cause effect
        self._action_check(prospective_location)
        did_win = self._goal_state_achived()
        
        self._agent_location = prospective_location

        reward = self._calc_reward()
        terminated = self.agent_dead or did_win
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _goal_state_achived(self):
        active_goal_states = {goal_state: False in self.goal_states for goal_state in self.goal_states if self.goal_states[goal_state]}
        
        if "no-dragon" in active_goal_states and self.dragon_location == (-1, -1):
            active_goal_states["no-dragon"] = True

        if "no-fire" in active_goal_states and len(self.fire_coords) == 0:
            active_goal_states["no-fire"] = True

        if "has-sword" in active_goal_states and self.has_sword == 1:
            active_goal_states["has-sword"] = True

        if False not in active_goal_states.values():
            return True

    def _action_check(self, location):
        # Check if action will give agent sword
        if np.array_equal(location, self.sword_location):
            self._print_debug("Agent picked up sword")
            self.has_sword = 1
            self.map[self.sword_location[0]][self.sword_location[1]].object = GameObject.EMPTY
            self.sword_location = (-1, -1)

        # Check if action will kill agent
        if np.array_equal(location, self.dragon_location) and self.has_sword == 0:
            self._print_debug("Agent tried to fight dragon without a sword")
            self._agent_location = (-1, -1)
            self.agent_dead = True

        # print(type(tuple(location)[0]))
        # fire_coords_array = np.array(self.fire_coords).astype(np.int32)
        for coord in self.fire_coords:
            if np.array_equal(tuple(location), coord):
                print("Agent died by walking into fire")
                self._agent_location = (-1, -1)
                self.agent_dead = True

        # TODO: Consider making ocean kill 
        if self.map[tuple(location)[0]][tuple(location)[1]].terrain == Terrain.OCEAN:
            self._print_debug("Agent died by walking into ocean")
            self._agent_location = (-1, -1)
            self.agent_dead = True

        # Check if action will kill dragon
        if np.array_equal(tuple(location), self.dragon_location) and self.has_sword == 1:
            self._print_debug("Agent killed dragon!")
            self.dragon_location = (-1, -1)
            self.map[self.dragon_location[0]][self.dragon_location[1]].object = GameObject.EMPTY

        # Check if action will kill fire
        if tuple(location) == self.mermaid_location:
            print("Agent talked to mermaid, she made it rain to put out the fire")
            for coord in self.fire_coords:
                self.map[coord[0]][coord[1]].object = GameObject.EMPTY
            self.fire_coords = []

        # Check if action will win game

    def _calc_reward(self):
        # Large reward for accomplishing goal state
        re_reward = 0
        if self._goal_state_achived():
            re_reward += 1000

        # TODO: For non-GPS agent, small reward for accomplishing sub-goal state

        # TODO: Small penalty for moving into impassable terrain
        
        # Penalty for death
        if self.agent_dead:
            re_reward -= 1000

        return re_reward

    def _load_sprites(self):
        # Load sprites
        self._agent_sprite = pygame.image.load(
            os.path.join(os.path.dirname(__file__), "static/Agent.png")
        )
        self._sword_sprite = pygame.image.load(
            os.path.join(os.path.dirname(__file__), "static/Sword.png")
        )
        self._dragon_sprite = pygame.image.load(
            os.path.join(os.path.dirname(__file__), "static/Dragon.png")
        )
        self._mermaid_sprite = pygame.image.load(
            os.path.join(os.path.dirname(__file__), "static/Mermaid.png")
        )

    

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
                if self.map[i][j].terrain == Terrain.OCEAN:
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 255),
                        pygame.Rect(
                            (i * pix_square_size, j * pix_square_size),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.map[i][j].terrain == Terrain.FOREST:
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        pygame.Rect(
                            (i * pix_square_size, j * pix_square_size),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.map[i][j].terrain == Terrain.MOUNTAIN:
                    pygame.draw.rect(
                        canvas,
                        (128, 128, 128),
                        pygame.Rect(
                            (i * pix_square_size, j * pix_square_size),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                if self.map[i][j].object == GameObject.FIRE:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            (i * pix_square_size, j * pix_square_size),
                            (pix_square_size, pix_square_size),
                        ),
                    ),
                elif self.map[i][j].object == GameObject.SWORD:
                    # Draw sword sprite
                    sword_sprite_scaled = pygame.transform.scale(self._sword_sprite, (pix_square_size, pix_square_size))
                    canvas.blit(
                        sword_sprite_scaled,
                        (i * pix_square_size, j * pix_square_size),
                    )
                # elif self.map[i][j] == Tiles.TREE_SAFE:
                #     pygame.draw.rect(
                #         canvas,
                #         (0, 255, 0),
                #         pygame.Rect(
                #             (i * pix_square_size, j * pix_square_size),
                #             (pix_square_size, pix_square_size),
                #         ),
                #     )
                # elif self.map[i][j] == Tiles.DRAGON:
                #     pygame.draw.rect(
                #         canvas,
                #         (255, 0, 255),
                #         pygame.Rect(
                #             (i * pix_square_size, j * pix_square_size),
                #             (pix_square_size, pix_square_size),
                #         ),
                #     )
                # elif self.map[i][j] == Tiles.SWORD:
                #     pygame.draw.rect(
                #         canvas,
                #         (255, 255, 0),
                #         pygame.Rect(
                #             (i * pix_square_size, j * pix_square_size),
                #             (pix_square_size, pix_square_size),
                #         )
                #     )


        # Draw dragon
        dragon_sprite_scaled = pygame.transform.scale(self._dragon_sprite, (pix_square_size, pix_square_size))
        canvas.blit(
            dragon_sprite_scaled,
            (self.dragon_location[0] * pix_square_size, self.dragon_location[1] * pix_square_size)
        )
        # Draw mermaid
        mermaid_sprite_scaled = pygame.transform.scale(self._mermaid_sprite, (pix_square_size, pix_square_size))
        canvas.blit(
            mermaid_sprite_scaled,
            (self.mermaid_location[0] * pix_square_size, self.mermaid_location[1] * pix_square_size)
        )
        # Now we draw the agent sprite
        agent_sprite_scaled = pygame.transform.scale(self._agent_sprite, (pix_square_size, pix_square_size))
        canvas.blit(
            agent_sprite_scaled,
            (self._agent_location[0] * pix_square_size, self._agent_location[1] * pix_square_size)
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )
        ## Render inventory box at bottom right corner
        pygame.draw.rect(
            canvas,
            (255, 255, 255),
            pygame.Rect(
                (self.window_size - 25, self.window_size - 25),
                (25, 25),
            ),
        )

        # Draw sword in inventory if agent has sword
        if self.has_sword == 1:
            pygame.draw.circle(
                canvas,
                (255, 255, 0),
                (self.window_size - 12.5, self.window_size - 12.5),
                12.5,
            )

        if self.render_mode == "human" and self.window is not None:
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
        
    def _print_debug(self, message):
        if self.metadata["debug"]:
            print(message)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
