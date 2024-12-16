import io
import re
import subprocess
import csv
from datetime import datetime
import gymnasium
import numpy as np
import torch
from village_safe.envs.env_enums import StartState, GoalState
import village_safe
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium.wrappers import TimeLimit
from stable_baselines3.a2c.policies import MultiInputPolicy
from EnvironmentStateClassifier import EnvironmentStateClassifier

import torch
import torch.nn as nn



def run_gps(world_state: str, goal_state: str):
    script_path = "GPS/run_gps.sh"
    output = subprocess.run(['wsl', 'bash', '-c', f"{script_path} \"{world_state}\" \"{goal_state}\""], capture_output=True, text=True)
    print(output.stdout)
    print(output.stderr)
    lines = output.stdout.split("\n")
    for line in lines:
        if line.startswith("((START)"):
            print(line)
            return line
    print(f"No start: {lines}")
    return None
        
def interpret_gps(goal_string):
    elements = re.findall(r'\((.*?)\)', goal_string)
    elements = elements[1:]

    # remove "EXECUTE" from each element
    elements = [element.split(" ")[1] for element in elements]

    return elements

def flatten_observation(observation):
    return np.concatenate([
        observation["object_map"].flatten(),
        observation["agent_loc"].flatten(),
        observation["mermaid_loc"].flatten(),
        observation["sword_loc"].flatten(),
        observation["dragon_loc"].flatten()
        ])

def decode_start_state(starting_state):
    match starting_state:
        case 0:
            return "dragon-alive sword-in-forest mermaid-in-ocean forest-on-fire"
        case 1: 
            return "dragon-alive sword-in-forest mermaid-on-beach forest-on-fire"
        case 2:
            return "dragon-alive sword-in-forest mermaid-on-beach forest-extinguished"
        case 3:
            return "dragon-alive has-sword mermaid-in-beach forest-extinguished"

def classify_start_state(model_path, obs):
    # load model
    with open(model_path, "rb") as file:
        buffer = io.BytesIO(file.read())
    input_size = 15*15 + 2 + 2 + 2 + 2 
    model = EnvironmentStateClassifier(input_size, 4)
    model = torch.load(buffer)
    model.eval()
    obs_flat = flatten_observation(obs)
    with torch.no_grad():
        obs_flat = torch.tensor(obs_flat).float().unsqueeze(0)
        output = model(obs_flat)
        return torch.argmax(output, dim=1).item()

def make_env():
    starting_state = np.random.randint(0, 4)
    boat_objective = mermaid_objective = sword_objective = True
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


# load model from "environment_classification_models\state_classifier_2.pth" and execute GPS
if __name__ == "__main__":
    # Find starting state
    env = make_env()
    obs, _ = env.reset()
    model_path = "environment_classification_models\state_classifier_2.pth"
    starting_state_encoded = classify_start_state(model_path, obs)
    print(f"Starting state: {starting_state_encoded}")
    
    start_state = decode_start_state(starting_state_encoded)
    
    # Execute GPS
    goal_state = "no-dragon"
    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")
    gps_output = run_gps(start_state, goal_state)
    interpreted_gps = interpret_gps(gps_output)
    print(f"GPS Output: {interpreted_gps}")


# if __name__ == "__main__":
#     #find starting state
#     env = make_env()
#     obs, _ = env.reset()
#     with open("environment_classification_models\state_classifier_2.pth", "rb") as file:
#         buffer = io.BytesIO(file.read())
#     input_size = 15*15 + 2 + 2 + 2 + 2 
#     model = EnvironmentStateClassifier(input_size, 4)
#     print(model.state_dict().keys())
#     model = torch.load(buffer)
#     # print(state_dict.keys())
#     # model.load_state_dict(state_dict, strict=False)
#     starting_state = classify_start_state(model, obs)
#     print(f"Starting state: {starting_state}")
#     goal_state = "no-dragon"

# model = PPO(MultiInputPolicy,
#             vec_env,
#             tensorboard_log=f"./village_safe_tensorboard/RandomState_PPO_{run_id}_lr_{ppo_hyperparameters['learning_rate']}_bs_{ppo_hyperparameters['batch_size']}",
#             **ppo_hyperparameters
#             )

# ppo_hyperparameters["run_id"] = run_id

# try:
#     with open("hyperparameters_record.csv", mode="x", newline="") as file:
#         writer = csv.DictWriter(file, fieldnames=ppo_hyperparameters.keys())
#         writer.writeheader()
# except FileExistsError:
#     pass  # File already exists, so we can append to it later

# with open ("hyperparameters_record.csv", mode="a", newline="") as file:
#     writer = csv.DictWriter(file, fieldnames=ppo_hyperparameters.keys())
#     writer.writerow(ppo_hyperparameters)

# print("Training model")
# model.learn(total_timesteps=4e6)

# model.save(f"RandomState_15x15_objectiveDistRewards_PPO_beachMermaid_{run_id}")

# vec_env.close()
