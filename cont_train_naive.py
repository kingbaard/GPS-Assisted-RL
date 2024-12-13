import csv
from datetime import datetime
import gymnasium
import village_safe
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium.wrappers import TimeLimit

# env = gymnasium.make("village_safe/VillageSafe-v0", render_mode="human", print={"actions": False, "rewards": True})

def make_env():
    env = gymnasium.make("village_safe/VillageSafe-v0")
    env = TimeLimit(env, max_episode_steps=100) 

    return env

vec_env = make_vec_env(make_env, n_envs=5)

# Use the vec_env wrapper to normalize rewards
# vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_obs=10.0)

run_id = datetime.now().strftime('%m%d_%H%M%S')
ppo_hyperparameters = {
    # "policy": "MultiInputPolicy",
    "n_epochs": 5,
    "batch_size": 256,
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "gamma": 0.95,
    "gae_lambda": 0.95,
    "clip_range": 0.25,
    "ent_coef": 0.1,
    "vf_coef": 0.6,
    "max_grad_norm": 0.5,
}

model = PPO.load("15x15_objectiveDistRewards_PPO_village_safe_1211_210225.zip",
            env=vec_env,
            verbose=1,
            tensorboard_log=f"./village_safe_tensorboard/{run_id}_lr_{ppo_hyperparameters['learning_rate']}_bs_{ppo_hyperparameters['batch_size']}",
            **ppo_hyperparameters
            )

model.learn(total_timesteps=1.5e6)

ppo_hyperparameters["run_id"] = run_id

try:
    with open("hyperparameters_record.csv", mode="x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ppo_hyperparameters.keys())
        writer.writeheader()
except FileExistsError:
    pass  # File already exists, so we can append to it later

with open ("hyperparameters_record.csv", mode="a", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=ppo_hyperparameters.keys())
    writer.writerow(ppo_hyperparameters)


print("Training model")
model.learn(total_timesteps=1.5e6)

model.save(f"continue_15x15_ODR_ppo_village_safe_{run_id}")

vec_env.close()

