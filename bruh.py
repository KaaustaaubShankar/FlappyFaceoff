import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

obs, _ = env.reset()
min_obs = [float('inf')] * 10
max_obs = [float('-inf')] * 10

while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    for i in [0,1,2,3,4,5,9]:
        min_obs[i] = min(min_obs[i], obs[i])
        max_obs[i] = max(max_obs[i], obs[i])
    
    # Checking if the player is still alive
    if terminated:
        break

print("Minimum of obs[0:7] and obs[9]: ", min_obs)
print("Maximum of obs[0:7] and obs[9]: ", max_obs)

env.close()
