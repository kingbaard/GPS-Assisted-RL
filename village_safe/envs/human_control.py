import gymnasium
import numpy as np
import pygame
from env_enums import Actions, StartState, GoalState
import village_safe

class HumanControl:

    def __init__(self):
        start_state = np.random.choice(list(StartState))
        self.env = gymnasium.make(
            "village_safe/VillageSafe-v0", 
            render_mode="human",
            goal_states={GoalState.IN_BOAT.value: False, GoalState.NO_FIRE.value: True, GoalState.HAS_SWORD.value: True, GoalState.NO_DRAGON.value: True},
            print={"actions": True, "rewards": True}
            )
        self.env.reset()
        self.running = True

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    self.handle_input(event)
                elif event.type == pygame.QUIT:
                    self.running = False
                    # self.close()

            # Render the environment after every interaction
            self.env.render()
            if self.env.agent_dead:
                self.running = False
                # self.close()

    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)
        if done:
            self.running = False
            print('Episode finished')
            self.env.close()
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed)
        self.env.render()

    def handle_input(self, event):
        key = event.key
        action = None

        if key == pygame.K_ESCAPE:  # Escape key to quit
            self.close()

        # Map key inputs to actions
        elif key == pygame.K_w:
            action = Actions.down
        elif key == pygame.K_d:
            action = Actions.right
        elif key == pygame.K_s:
            action = Actions.up
        elif key == pygame.K_a:
            action = Actions.left

        # If a valid action is mapped, execute a step
        if action is not None:
            self.step(action.value)

    def close(self):
        self.env.close()


if __name__ == "__main__":
    human_control = HumanControl()
    human_control.run()
    human_control.close()