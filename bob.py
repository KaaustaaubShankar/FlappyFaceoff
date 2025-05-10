import gymnasium as gym
import pygame
import matplotlib.pyplot as plt
import time
from gym.wrappers import RecordVideo
from datetime import datetime
import flappy_bird_gymnasium


click_times = []

def manual_play_and_record():
    global click_times
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False, normalize_obs=True)

    observation, _ = env.reset(seed=42)
    total_reward = 0

    observation_times = []
    observations = []
    score_times = []
    last_score = 0

    start_time = time.time()

    try:
        while True:
            action = 0  # Default: do nothing

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    plot_selected_inputs(observation_times, observations, score_times)
                    return total_reward
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    action = 1  # Space means flap
                    click_times.append(time.time() - start_time)

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            current_time = time.time() - start_time
            observation_times.append(current_time)
            observations.append(observation)

            current_score = info.get('score', 0)
            if current_score > last_score:
                score_times.append((current_time, current_score))
                last_score = current_score

            if terminated or truncated:
                break

    finally:
        env.close()
        plot_selected_inputs(observation_times, observations, score_times)

    return total_reward


def plot_selected_inputs(times, observations, score_times):
    global click_times

    selected_observations = []
    for obs in observations:
        selected = [
            obs[0],
            (obs[2] + obs[1]) / 2,
            obs[3],
            obs[4] + 50,
            obs[7] + 50,
            obs[9]
        ]
        selected_observations.append(selected)

    selected_observations = list(zip(*selected_observations))

    labels = [
        "Last Pipe X",
        "Middle of Last Gap Y",
        "Next Pipe X",
        "Next Top Pipe Y + 50",
        "Next Next Top Pipe Y + 50",
        "Player Y"
    ]

    fig, axs = plt.subplots(6, 1, figsize=(15, 18), sharex=True)

    for i, (ax, obs) in enumerate(zip(axs, selected_observations)):
        ax.plot(times, obs)
        for idx, (score_time, score) in enumerate(score_times):
            ax.axvline(x=score_time, color='red', linestyle='--')
            ax.text(score_time, ax.get_ylim()[1], f'Score {score}', color='red', rotation=90, verticalalignment='bottom', horizontalalignment='right', fontsize=8)
        for click_time in click_times:
            ax.axvline(x=click_time, color='orange', linestyle='--')
        ax.set_ylabel(labels[i])

    axs[-1].set_xlabel('Time (s)')
    plt.suptitle('Selected Observations Over Time with Score and Click Events')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    manual_play_and_record()
