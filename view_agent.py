import csv
from datetime import datetime
import gymnasium
from village_safe.envs.env_enums import GoalState
import village_safe
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium.wrappers import TimeLimit
from stable_baselines3.a2c.policies import MultiInputPolicy
# from custom_network import CustomActorCriticPolicy

def make_env():
    env = gymnasium.make(
        "village_safe/VillageSafe-v0", 
        render_mode="human",
        goal_states={GoalState.IN_BOAT.value: False, GoalState.NO_FIRE.value: True, GoalState.HAS_SWORD.value: True, GoalState.NO_DRAGON.value: True},
        )
    env = TimeLimit(env, max_episode_steps=100) 

    return env

env = make_env()

# vec_env = make_vec_env(make_env, n_envs=1)

model = PPO.load("LargerNet_15x15_objectiveDistRewards_PPO_village_safe_1213_154057.zip",
            env=env,
            verbose=1
            )

cumulative_reward_hist = []
episode_length_hist = []

for episode in range(20):
    obs, _ = env.reset()
    done = False
    cumulative_reward = 0
    episode_length = 0
    while not done:
        # Get action from the model
        action, _states = model.predict(obs)
        # print(action)
        # Step through the environment
        obs, rewards, dones, _, info = env.step(int(action))
        print(rewards)
        
        # Render only the first environment
        env.render()
        cumulative_reward += rewards
        episode_length += 1

        if dones:
            cumulative_reward_hist.append(cumulative_reward)
            episode_length_hist.append(episode_length)
            break

env.close()
average_reward = sum(cumulative_reward_hist) / len(cumulative_reward_hist)
average_episode_length = sum(episode_length_hist) / len(episode_length_hist)

print(f"Average reward: {average_reward}")
print(f"Average episode length: {average_episode_length}")