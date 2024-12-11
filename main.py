import gymnasium
import village_safe

env = gymnasium.make("village_safe/VillageSafe-v0", render_mode="human", print={"actions": True, "rewards": True})

for episode in range(10):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Take a random action
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # End the episode if done
        if terminated or truncated:
            break

env.close()