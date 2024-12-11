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
    env = TimeLimit(env, max_episode_steps=500) 

    return env

vec_env = make_vec_env(make_env, n_envs=4)

# Use the vec_env wrapper to normalize rewards
# vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_obs=10.0)

run_id = datetime.now().strftime('%m%d_%H%M%S')
hyperparameters = {
    # "policy": "MultiInputPolicy",
    "n_epochs": 10,
    "batch_size": 128,
    "learning_rate": 2e-4,
    "n_steps": 512,
    "gamma": 0.98,
    "gae_lambda": 0.94,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

model = PPO("MultiInputPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=f"./village_safe_tensorboard/{run_id}_lr_{hyperparameters['learning_rate']}_bs_{hyperparameters['batch_size']}",
            **hyperparameters
            )

hyperparameters["run_id"] = run_id

try:
    with open("hyperparameters_record.csv", mode="x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=hyperparameters.keys())
        writer.writeheader()
except FileExistsError:
    pass  # File already exists, so we can append to it later

with open ("hyperparameters_record.csv", mode="a", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=hyperparameters.keys())
    writer.writerow(hyperparameters)


print("Training model")
model.learn(total_timesteps=1e6)

model.save(f"ppo_village_safe_{run_id}")

vec_env.close()

