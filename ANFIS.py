import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import flappy_bird_gymnasium
import collections
import matplotlib.pyplot as plt

# --- Part 1: ANFIS Model that predicts two Q-values ---

class GaussianMF:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def eval(self, x):
        return np.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)

class ExtendedANFIS(nn.Module):
    def __init__(self, num_inputs=7, num_mfs=3, num_outputs=2):
        super(ExtendedANFIS, self).__init__()
        self.num_inputs = num_inputs
        self.num_mfs = num_mfs
        self.num_outputs = num_outputs
        self.mfs = nn.Parameter(torch.randn(num_inputs, num_mfs, 2))  # mean and sigma
        self.rules = nn.Parameter(torch.randn(num_mfs ** num_inputs, num_outputs))

    def fuzzify(self, x):
        memberships = []
        for i in range(self.num_inputs):
            means = self.mfs[i, :, 0]
            sigmas = torch.abs(self.mfs[i, :, 1]) + 1e-6
            evals = torch.exp(-0.5 * ((x[i] - means) / sigmas) ** 2)
            memberships.append(evals)
        return memberships

    def forward(self, x):
        batch_size = x.size(0)
        memberships = [self.fuzzify(x[i]) for i in range(batch_size)]

        outputs = []
        for b in range(batch_size):
            firing_strengths = []
            for idx in range(self.rules.size(0)):
                idxs = np.unravel_index(idx, (self.num_mfs,) * self.num_inputs)
                strength = torch.prod(torch.stack([memberships[b][i][idxs[i]] for i in range(self.num_inputs)]))
                firing_strengths.append(strength)
            firing_strengths = torch.stack(firing_strengths)
            normalized_strengths = firing_strengths / (firing_strengths.sum() + 1e-6)
            output = torch.matmul(normalized_strengths, self.rules)
            outputs.append(output)

        return torch.stack(outputs)

# --- Part 2: Replay Buffer ---

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)

# --- Part 3: DQN Agent ---

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ExtendedANFIS().to(self.device)
        self.target_model = ExtendedANFIS().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer()
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return q_values.argmax().item()

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

# --- Part 4: Training Loop ---

def train_dqn():
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False, normalize_obs=True)
    agent = DQNAgent()
    num_episodes = 500
    scores = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=random.randint(0, 10000))
        state = np.array([state[6], state[7], state[8], state[3], state[4], state[5], state[9]])
        total_reward = 0

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array([next_state[6], next_state[7], next_state[8], next_state[3], next_state[4], next_state[5], next_state[9]])
            done = terminated or truncated
            agent.buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            agent.train()

        agent.update_target()
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        scores.append(total_reward)

        print(f"Episode {episode+1}: Total Reward = {total_reward}")

    env.close()

    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("ANFIS-DQN Flappy Bird Training")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_dqn()
