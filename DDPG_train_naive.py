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

# env = gymnasium.make("village_safe/VillageSafe-v0", render_mode="human", print={"actions": False, "rewards": True})

larger_arc = [dict(pi=[256, 256], vf=[256, 256])]

def make_env():
    env = gymnasium.make("village_safe/VillageSafe-v0")
    env = TimeLimit(env, max_episode_steps=750) 

    return env

vec_env = make_vec_env(make_env, n_envs=300)

# Use the vec_env wrapper to normalize rewards
# vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_obs=10.0)

run_id = datetime.now().strftime('%m%d_%H%M%S')
ppo_hyperparameters = {
    # "policy": "MultiInputPolicy",
    "n_epochs": 10,
    "batch_size": 256,
    "learning_rate": 3e-4,
    "n_steps": 512,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.03,
    "vf_coef": 0.6,
    "max_grad_norm": 0.5,
}

sac_hyperparameters = {
        "learning_rate":3e-4,
        "buffer_size": 1e6,
        "batch_size": 256,
        "train_freq": 1,  # Update every step
        "gradient_steps": 1,  # One gradient step per training iteration
        "tau": 0.005,  # Target network smoothing
        "gamma": 0.99,  # Discount factor
        'policy_kwargs': dict(net_arch=[256, 256]),  # Larger network for complex environments
        'verbose': 1,
    }

model = SAC(MultiInputPolicy,
            vec_env,
            tensorboard_log=f"./village_safe_tensorboard/SAC_{run_id}_lr_{ppo_hyperparameters['learning_rate']}_bs_{ppo_hyperparameters['batch_size']}",
            **sac_hyperparameters
            )

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

model.save(f"SAC_village_safe_{run_id}")

vec_env.close()

