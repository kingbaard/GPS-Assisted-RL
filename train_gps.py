import re
import subprocess
import csv
from datetime import datetime
import gymnasium
import numpy as np
from village_safe.envs.env_enums import StartState, GoalState
import village_safe
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium.wrappers import TimeLimit
from stable_baselines3.a2c.policies import MultiInputPolicy

def run_gps(goal_state: str):
    script_path = "GPS/run_gps.sh"
    output = subprocess.run(['wsl', 'bash', '-c', f"{script_path} {goal_state}"], capture_output=True, text=True)
    
    lines = output.stdout.split("\n")
    for line in lines:
        if line.startswith("((START)"):
            return line
        
def interpret_gps(goal_string):
    elements = re.findall(r'\((.*?)\)', goal_string)
    elements = elements[1:]

    return elements

def make_env():
    starting_state = np.randint(0, 4)
    boat_objective, mermaid_objective, sword_objective = True
    if starting_state >= 1:
        boat_objective = False
    if starting_state >= 2:
        mermaid_objective = False
    if starting_state >= 3:
        sword_objective = False

    env = gymnasium.make(
        "village_safe/VillageSafe-v0", 
        goal_states={GoalState.IN_BOAT.value: boat_objective, GoalState.NO_FIRE.value: mermaid_objective, GoalState.HAS_SWORD.value: sword_objective, GoalState.NO_DRAGON.value: True},
        )
    env = TimeLimit(env, max_episode_steps=100) 

    return env

vec_env = make_vec_env(make_env, n_envs=8)
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

model = PPO(MultiInputPolicy,
            vec_env,
            tensorboard_log=f"./village_safe_tensorboard/BeachMermaid_PPO_{run_id}_lr_{ppo_hyperparameters['learning_rate']}_bs_{ppo_hyperparameters['batch_size']}",
            **ppo_hyperparameters
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
model.learn(total_timesteps=5e6)

model.save(f"15x15_objectiveDistRewards_PPO_beachMermaid_{run_id}")

vec_env.close()
