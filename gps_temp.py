import re
import subprocess

def run_gps(world_state: str, goal_state: str):
    script_path = "./GPS/run_gps.sh"
    output = subprocess.run(['wsl', 'bash', '-c', f"{script_path} {world_state} {goal_state}"], capture_output=True, text=True)
    print(output)

    lines = output.stdout.split("\n")
    for line in lines:
        if line.startswith("((START)"):
            return line
        
def interpret_gps(goal_string):
    print(f"GOAL: {goal_string}")
    elements = re.findall(r'\((.*?)\)', goal_string)
    elements = elements[1:]

    return elements

if __name__ == "__main__": 
    world_state = "off-boat mermaid-on-beach forest-on-fire"
    goal_state = "NO_DRAGON"
    goal_string = run_gps(world_state, goal_state)
    elements = interpret_gps(goal_string)
    print(elements)