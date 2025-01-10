import numpy as np
import random
import pickle
from app import FlappyBirdEnv  # Import the environment

class QLearningAgent:
    def __init__(self, action_space, state_space, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999, bins=10):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.action_space = action_space
        self.state_space = state_space
        self.bins = bins  # Number of bins to discretize continuous state space
        
        # Discretize the state space into bins
        self.state_bounds = list(zip(self.state_space.low, self.state_space.high))
        self.state_bins = [np.linspace(low, high, bins) for low, high in self.state_bounds]
        
        # Initialize the Q-table for discretized states using a dictionary
        self.Q = {}

    def discretize_state(self, state):
        """Convert a continuous state into a discrete index."""
        state_index = []
        for i, value in enumerate(state):
            # Find the index of the value in the corresponding bin
            state_index.append(np.digitize(value, self.state_bins[i]) - 1)
        return tuple(state_index)

    def get_action(self, state):
        state_index = self.discretize_state(state)
        
        # If the state is not in the Q-table, initialize it with zeros
        if state_index not in self.Q:
            self.Q[state_index] = np.zeros(self.action_space.n)
        
        if random.uniform(0, 1) < self.epsilon:  # Exploration
            return self.action_space.sample()
        else:  # Exploitation
            return np.argmax(self.Q[state_index])  # Return the action with the highest Q-value

    def update(self, state, action, reward, next_state):
        state_index = self.discretize_state(state)
        next_state_index = self.discretize_state(next_state)
        
        # Initialize Q-values if the state or next_state is not in the Q-table
        if state_index not in self.Q:
            self.Q[state_index] = np.zeros(self.action_space.n)
        if next_state_index not in self.Q:
            self.Q[next_state_index] = np.zeros(self.action_space.n)
        
        best_next_action = np.argmax(self.Q[next_state_index])  # Max Q-value for next state
        # Q-learning update rule
        self.Q[state_index][action] += self.alpha * (reward + self.gamma * self.Q[next_state_index][best_next_action] - self.Q[state_index][action])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        """Save the Q-table to a file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.Q, f)
    
    def load(self, filename):
        """Load the Q-table from a file"""
        with open(filename, 'rb') as f:
            self.Q = pickle.load(f)

def train():
    env = FlappyBirdEnv(render_mode=False)  # Disable rendering during training
    agent = QLearningAgent(env.action_space, state_space=env.observation_space)

    max_reward = -float('inf')  
    min_reward = float('inf')  # Initialize min_reward to a very large value
    max_score = 0  
    total_rewards = 0
    rewards_list = []  

    for episode in range(20000):  # Number of episodes
        state = env.reset()
        done = False
        episode_reward = 0  
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            episode_reward += reward  # Accumulate reward for the current episode
        
        total_rewards += episode_reward
        rewards_list.append(episode_reward)
        # Update the maximum and minimum rewards
        if episode_reward >= max_reward:
            max_reward = episode_reward
            max_score = env.score  # Store the score of the episode with the max reward
            # Save the best performing agent
            agent.save('best_agent.pkl')
        if episode_reward < min_reward:
            min_reward = episode_reward
        
        if episode % 100 == 0:  # Log every 100 episodes
            avg_reward = total_rewards / (episode + 1)
            percentile_25 = np.percentile(rewards_list, 25)
            percentile_75 = np.percentile(rewards_list, 75)
            print(f"Episode {episode}: Max Reward {max_reward}, Min Reward {min_reward}")
            print(f"25th Percentile: {percentile_25}, 75th Percentile: {percentile_75}")
            print(f"Average Reward {avg_reward}")
            print(f"Current Epsilon: {agent.epsilon:.4f}")

    print(f"Training completed. Max Reward: {max_reward} at score {max_score}")
    print(f"Min Reward: {min_reward}")
    print(f"25th Percentile: {np.percentile(rewards_list, 25)}")
    print(f"75th Percentile: {np.percentile(rewards_list, 75)}")
    env.close()

def play():
    """Watch the trained agent play"""
    env = FlappyBirdEnv(render_mode=True)  # Enable rendering
    agent = QLearningAgent(env.action_space, state_space=env.observation_space)
    agent.load('best_agent.pkl')  # Load the best trained agent
    agent.epsilon = 0  # No exploration during play
    
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.get_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
    
    print(f"Game Over! Final Score: {env.score}, Total Reward: {total_reward}")
    env.close()

if __name__ == '__main__':
    train()
    play()
