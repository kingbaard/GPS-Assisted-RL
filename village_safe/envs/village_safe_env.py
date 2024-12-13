import sys
import os

from .env_enums import Actions, Zone, Terrain, GameObject, ObservationMap, ObjectiveStatus, GoalState, StartState
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from .WorldBuilder import WorldBuilder

class VillageSafeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
        "goal_state": GoalState.NO_DRAGON,} # Will be used to train indiviual actions for GPS-assisted agent

    def __init__(
            self, 
            render_mode=None, 
            start_state=StartState.BOAT, 
            goal_states={GoalState.IN_BOAT: False, GoalState.NO_FIRE: False, GoalState.HAS_SWORD: False, GoalState.NO_DRAGON: True}, 
            control_mode=None,
            print={"actions": False, "rewards": False}
            ):
        self.size = 15  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.world_builder = WorldBuilder(self.size, 0, start_state=start_state)
        self.world_builder.generate_world()
        self.map = self.world_builder.map
        self.boat_location, self.mermaid_location, self.sword_location, self.dragon_location, self.fire_coords = self.world_builder.return_object_coords()
        self.has_sword = 0
        self.agent_dead = False
        self.print = print
        self.in_boat = False

        self.start_state = start_state
        self.goal_states = goal_states
        self.current_objective = self._determine_objective()

        self.objective_states = {
                                GoalState.IN_BOAT: ObjectiveStatus.NOT_REACHED,
                                GoalState.NO_FIRE: ObjectiveStatus.NOT_REACHED,
                                GoalState.HAS_SWORD: ObjectiveStatus.NOT_REACHED,
                                GoalState.NO_DRAGON: ObjectiveStatus.NOT_REACHED,
                                }
        self.steps_since_completion = {
                                GoalState.IN_BOAT: 0,
                                GoalState.NO_FIRE: 0,
                                GoalState.HAS_SWORD: 0,
                                GoalState.NO_DRAGON: 0,
                                }

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent_loc": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                # "sword_loc": spaces.Box(-1, self.size - 1, shape=(2,), dtype=int),
                "dragon_loc": spaces.Box(-1, self.size - 1, shape=(2,), dtype=int),
                "mermaid_loc": spaces.Box(-1, self.size - 1, shape=(2,), dtype=int),
                # "object_map": spaces.Box(0, 5, shape=(self.size, self.size), dtype=int), # 2d
                "object_map": spaces.Box(0, 5, shape=(self.size * self.size,), dtype=int), # 2d
                # "has-sword": spaces.Discrete(2), # 0 if agent does not have sword, 1 if agent has sword  
            }
        )

        # Discrete action-space
        self.action_space = spaces.Discrete(4) # "right", "up", "left", "down", "right"

        # Box action-space
        # self.action_space = spaces.Box(0, 4, shape=(1,), dtype=int)

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

        self.window = None
        self.clock = None

    def _get_obs(self):
        object_map = np.zeros((self.size, self.size)).astype(np.int32)
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i][j].object == GameObject.FIRE:
                    object_map[i][j] = ObservationMap.FIRE.value
                elif self.map[i][j].object == GameObject.SWORD:
                    object_map[i][j] = ObservationMap.SWORD.value
                elif self.map[i][j].object == GameObject.ROCK:
                    object_map[i][j] = ObservationMap.ROCK.value
                elif self.map[i][j].terrain == Terrain.OCEAN:
                    object_map[i][j] = ObservationMap.WATER.value
                else:
                    object_map[i][j] = ObservationMap.NOTHING.value
        return {
            "agent_loc": np.array(self._agent_location).astype(np.int32),
            # "sword_loc": np.array(self.sword_location).astype(np.int32),
            "mermaid_loc": np.array(self.mermaid_location).astype(np.int32), 
            "dragon_loc": np.array(self.dragon_location).astype(np.int32),
            "object_map": object_map.flatten(),
            # "has-sword": self.has_sword,
        }

    def _get_info(self):
        return {
        }
    

    def _randomize_world_state(self):
        random_state = np.random.choice(StartState.BOAT, StartState.MERMAID, StartState.SWORD, StartState.DRAGON)

        if random_state in [StartState.SWORD, StartState.DRAGON]:
            self.objective_states["no-fire"] = ObjectiveStatus.REACHED_PAST
        elif random_state == StartState.DRAGON:
            self.objective_states["has-sword"] = ObjectiveStatus.REACHED_PAST
        # TODO: more advanced


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(self.size // 3, self.size // 3 * 2, size=2, dtype=int)

        self.world_builder.start_state = self.start_state
        self.world_builder.generate_world()
        self.map = self.world_builder.map
        self.boat_location, self.mermaid_location, self.sword_location, self.dragon_location, self.fire_coords = self.world_builder.return_object_coords()
        self.on_boat = False
        self.has_sword = 0
        self.agent_dead = False
        observation = self._get_obs()
        info = self._get_info()

        self.current_objective = self._determine_objective()

        if self.render_mode == "human":
            self._render_frame()

        self.objective_states = {
                                "no-fire": ObjectiveStatus.NOT_REACHED,
                                "has-sword": ObjectiveStatus.NOT_REACHED,
                                "no-dragon": ObjectiveStatus.NOT_REACHED,
                                }
        self.steps_since_completion = {
                                "no-fire": 0,
                                "has-sword": 0,
                                "no-dragon": 0,
                                }

        return observation, info


    def step(self, action):
        direction = self._action_to_direction[action]

        # ensure agent doesnt walk out of bounds
        prospective_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        valid_action = self._action_check(prospective_location)
        did_win = self._goal_state_achived()
        self._update_objective_state()
        
        reward = self._calc_reward(valid_action, prospective_location)
        
        if valid_action:
            self._agent_location = prospective_location

            if self.in_boat:
                self.boat_location = prospective_location

        terminated = self.agent_dead or did_win
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _update_objective_state(self):
        for objective in self.objective_states.keys():
            if self.objective_states[objective] == ObjectiveStatus.REACHED_PAST:
                self.steps_since_completion[objective] += 1
                continue
            # Set new current objective
            if self.objective_states[objective] == ObjectiveStatus.REACHED_THIS_FRAME:
                self.objective_states[objective] = ObjectiveStatus.REACHED_PAST
                if objective == GoalState.NO_DRAGON:
                    self.goal_state = GoalState.HAS_SWORD
                elif objective == "has-sword":
                    self.goal_state = GoalState.NO_DRAGON
                continue
            if objective == "no-dragon" and self.dragon_location == (-1, -1):
                self.objective_states["no-dragon"] = ObjectiveStatus.REACHED_THIS_FRAME
            elif objective == "has-sword" and self.has_sword == 1:
                self.objective_states["has-sword"] = ObjectiveStatus.REACHED_THIS_FRAME
            elif objective == "no-fire" and len(self.fire_coords) == 0:
                self.objective_states["no-fire"] = ObjectiveStatus.REACHED_THIS_FRAME
            elif objective == 


    def _goal_state_achived(self):
        # active_goal_states = {state: False in state for state in self.objective_states if self.objective_states[state]}
        
        if GoalState.NO_DRAGON in self.goal_states and self.dragon_location == (-1, -1):
            self.goal_states[GoalState.NO_DRAGON] = True
        if GoalState.HAS_SWORD in self.goal_states and self.has_sword == 1:
            self.goal_states[GoalState.HAS_SWORD] = True
        if GoalState.NO_FIRE in self.goal_states and len(self.fire_coords) == 0:
            self.goal_states[GoalState.NO_FIRE] = True
        if GoalState.IN_BOAT in self.goal_states and (self.in_boat or len(self.fire_coords) == 0):
            self.goal_states[GoalState.IN_BOAT] = True
        if False not in self.goal_states.values():
            return True
        return False

    def _action_check(self, location):
        # check if action will move agent into impassable terrain
        if self.map[location[0]][location[1]].object == GameObject.ROCK: 
            self._print_actions_debug("Agent tried to walk into rock")
            return False
        
        # check if action will move agent out of bounds
        if np.array_equal(location, self._agent_location):
            return False

        # Check to see if agent is entering boat
        if np.array_equal(location, self.boat_location):
            self._print_actions_debug("Agent entered boat")
            self.in_boat = True

        # Check to see if agent is leaving boat
        if self.in_boat and self.map[location[0]][location[1]].terrain not in [Terrain.BEACH, Terrain.OCEAN]:
            self._print_actions_debug("Agent left boat")
            self.in_boat = False
        
        # Check if action will give agent sword
        if np.array_equal(location, self.sword_location):
            self._print_actions_debug("Agent picked up sword")
            self.has_sword = 1
            self.map[self.sword_location[0]][self.sword_location[1]].object = GameObject.EMPTY
            self.sword_location = (-1, -1)

        # Check if action will kill agent
        if np.array_equal(location, self.dragon_location) and self.has_sword == 0:
            self._print_actions_debug("Agent tried to fight dragon without a sword")
            self._agent_location = (-1, -1)
            self.agent_dead = True
        
        # print(type(tuple(location)[0]))
        # fire_coords_array = np.array(self.fire_coords).astype(np.int32)
        for coord in self.fire_coords:
            if np.array_equal(tuple(location), coord):
                self._print_actions_debug("Agent died by walking into fire")
                self._agent_location = (-1, -1)
                self.agent_dead = True

        # Player will die if they walk into ocean 
        if self.map[tuple(location)[0]][tuple(location)[1]].terrain == Terrain.OCEAN and not self.in_boat:
            self._print_actions_debug("Agent died by walking into ocean")
            self._agent_location = (-1, -1)
            self.agent_dead = True

        # Check if action will kill dragon
        if np.array_equal(tuple(location), self.dragon_location) and self.has_sword == 1:
            self._print_actions_debug("Agent killed dragon!")
            self.dragon_location = (-1, -1)
            self.map[self.dragon_location[0]][self.dragon_location[1]].object = GameObject.EMPTY

        # Check if action will kill fire
        if tuple(location) == self.mermaid_location:
            self._print_actions_debug("Agent talked to mermaid, she made it rain to put out the fire")
            for coord in self.fire_coords:
                self.map[coord[0]][coord[1]].object = GameObject.EMPTY
            self.fire_coords = []
        return True
    
    def _determine_objective(self):
        if self.goal_states[GoalState.IN_BOAT] and self.objective_states[GoalState.IN_BOAT] == ObjectiveStatus.NOT_REACHED:
            return GoalState.IN_BOAT
        elif self.goal_states[GoalState.NO_FIRE] and self.objective_states[GoalState.NO_FIRE] == ObjectiveStatus.NOT_REACHED:
            return GoalState.NO_FIRE
        elif self.goal_states[GoalState.HAS_SWORD] and self.objective_states[GoalState.HAS_SWORD] == ObjectiveStatus.NOT_REACHED:
            return GoalState.HAS_SWORD
        return GoalState.NO_DRAGON

    def _calc_reward(self, valid_action, prospective_location):
        re_reward = 0

        # Large reward for accomplishing goal state
        if self._goal_state_achived():
            re_reward += 100

        # Small reward for moving towards current objective
        re_reward += self._reward_objective_proximity(prospective_location=prospective_location)
        
        # Small reward for accomplishing sub-goal state
        # reward for putting out fire
        reward_smooth_factor = 1
        if self.objective_states["no-fire"] == ObjectiveStatus.REACHED_THIS_FRAME:
            print("Giving reward for putting out fire")
            re_reward += 50
        elif self.objective_states["no-fire"] == ObjectiveStatus.REACHED_PAST:
            re_reward += max(10 - (self.steps_since_completion["no-fire"] * reward_smooth_factor), 0)

        # reward for picking up sword
        if self.objective_states["has-sword"] == ObjectiveStatus.REACHED_THIS_FRAME:
            print("Giving reward for picking up sword")
            re_reward += 50
        elif self.objective_states["has-sword"] == ObjectiveStatus.REACHED_PAST:
            re_reward += max(10 - (self.steps_since_completion["has-sword"] * reward_smooth_factor), 0)

        # reward for killing dragon
        if self.objective_states["no-dragon"] == ObjectiveStatus.REACHED_THIS_FRAME:
            print("Giving reward for killing dragon")
            re_reward += 50
        elif self.objective_states["no-dragon"] == ObjectiveStatus.REACHED_PAST:
            re_reward += max(10 - (self.steps_since_completion["no-dragon"] * reward_smooth_factor), 0)

        # Small penalty for attempting to move into impassable terrain
        if not valid_action:
            re_reward -= 5
        
        # Penalty for death
        if self.agent_dead:
            re_reward -= 50

        # Normalize reward such that it is in the range [-1, 1]
        # re_reward = np.clip(re_reward, -100, 100)
        # reward_max = 100
        # reward_min = -100
        # re_reward = (re_reward - reward_min) / (reward_max - reward_min) * 2 - 1

        # re_reward = round(re_reward, 5) # round to 5 decimal places to avoid floating point errors

        if self.print["rewards"]:
            print(f"Reward: {re_reward}")
        return re_reward

    def _reward_objective_proximity(self, prospective_location):
        objective_location = None
        if self.current_objective == GoalState.NO_FIRE:
            objective_location = self.mermaid_location
        elif self.current_objective == GoalState.HAS_SWORD:
            objective_location = self.sword_location
        elif self.current_objective == GoalState.NO_DRAGON:
            objective_location = self.dragon_location
            # reward moving towards dragon
        past_distance = np.linalg.norm(np.array(self._agent_location) - np.array(objective_location)) 
        new_distance =  np.linalg.norm(np.array(prospective_location) - np.array(objective_location))

        distance_delta = past_distance - new_distance
        print(f"Distance delta: {distance_delta}")
        if distance_delta > 0.75:
            return 3
        elif distance_delta < 0:
            return -3
        return 0
    
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
        self._boat_sprite = pygame.image.load(
            os.path.join(os.path.dirname(__file__), "static/Boat.png")
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
                elif self.map[i][j].terrain == Terrain.BEACH:
                    pygame.draw.rect(
                        canvas,
                        (173, 151, 42),
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
                elif self.map[i][j].object == GameObject.ROCK:
                    pygame.draw.rect(
                        canvas,
                        (200, 200, 200),
                        pygame.Rect(
                            (i * pix_square_size, j * pix_square_size),
                            (pix_square_size, pix_square_size),
                        ),
                    )
            

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
        # Draw boat
        boat_sprite_scaled = pygame.transform.scale(self._boat_sprite, (pix_square_size, pix_square_size))
        canvas.blit(
            boat_sprite_scaled,
            (self.boat_location[0] * pix_square_size, self.boat_location[1] * pix_square_size)
        )
        # Now we draw the agent sprite
        if not self.in_boat:
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

    def _print_actions_debug(self, message):
        if self.print["actions"]:
            print(message)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
