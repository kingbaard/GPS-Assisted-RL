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
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


def make_env():
    env = gymnasium.make(
        "village_safe/VillageSafe-v0", 
        render_mode="rgb_array",
        goal_states={GoalState.IN_BOAT.value: False, GoalState.NO_FIRE.value: True, GoalState.HAS_SWORD.value: True, GoalState.NO_DRAGON.value: True},
        )
    env = TimeLimit(env, max_episode_steps=100) 
    env = RecordVideo(env, video_folder="./recordings", name_prefix="video", disable_logger=True)

    return env

env = make_env()

policy_paths = {
    "on_boat": None,
    "no_fire_beach": PPO.load("15x15_objectiveDistRewards_PPO_beachMermaid_1213_145948.zip", env=env),
    "no_fire_ocean": None,
    "has_sword": PPO.load("15x15_objectiveDistRewards_PPO_get_sword_1213_124856.zip", env=env),
    "no_dragon": PPO.load("15x15_objectiveDistRewards_PPO_dragon_1213_133112.zip", env=env)
}

def choose_policy(current_objective):
    match current_objective:
        case GoalState.IN_BOAT:
            return policy_paths["on_boat"]
        case GoalState.NO_FIRE:
            return policy_paths["no_fire_beach"]
        case GoalState.HAS_SWORD:
            return policy_paths["has_sword"]
        case GoalState.NO_DRAGON:
            return policy_paths["no_dragon"]
cumulative_reward_hist = []
episode_length_hist = []
def run_episode():
    obs, info = env.reset()
    env.start_video_recorder()
    done = False
    cumulative_reward = 0
    episode_length = 0
    while not done:
        model = choose_policy(info["current_objective"])
        action, _states = model.predict(obs)
        # print(action)
        # Step through the environment
        obs, rewards, done, _, info = env.step(int(action))
        cumulative_reward += rewards
        episode_length += 1
        if done:
            cumulative_reward_hist.append(cumulative_reward)
            episode_length_hist.append(episode_length)
            env.close_video_recorder()
            break

if __name__ == "__main__":
    for episode in range(20):
        run_episode()
        env.close()

    average_reward = sum(cumulative_reward_hist) / len(cumulative_reward_hist)
    average_episode_length = sum(episode_length_hist) / len(episode_length_hist)

    print(f"Average reward: {average_reward}")
    print(f"Average episode length: {average_episode_length}")