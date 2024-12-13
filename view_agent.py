import csv
from datetime import datetime
import gymnasium
import village_safe
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium.wrappers import TimeLimit
from stable_baselines3.a2c.policies import MultiInputPolicy
# from custom_network import CustomActorCriticPolicy

env = gymnasium.make("village_safe/VillageSafe-v0", render_mode="human", print={"actions": True, "rewards": True})

def make_env():
    env = gymnasium.make("village_safe/VillageSafe-v0",  render_mode="human", print={"actions": True, "rewards": False})
    env = TimeLimit(env, max_episode_steps=100) 

    return env

env = make_env()

# vec_env = make_vec_env(make_env, n_envs=1)

model = PPO.load("15x15_PPO_village_safe_1211_195225.zip",
            env=env,
            verbose=1
            )

for episode in range(10):
    obs, _ = env.reset()
    done = False
    while not done:
        # Get action from the model
        action, _states = model.predict(obs)
        # print(action)
        # Step through the environment
        obs, rewards, dones, _, info = env.step(int(action))
        print(rewards)
        
        # Render only the first environment
        env.render()

        if dones:
            break

env.close()