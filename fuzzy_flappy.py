import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import matplotlib.pyplot as plt
import pygame
import imageio
import os
from datetime import datetime

# Global GA parameters
POP_SIZE = 200
N_GENERATIONS = 15
TOURNAMENT_SIZE = 3
CX_PROB = 0.7
MUT_PROB = 0.2

class SeedWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._seed = seed

    def reset(self, **kwargs):
        # Use stored seed unless explicitly provided
        if 'seed' not in kwargs:
            kwargs['seed'] = self._seed
        return super().reset(**kwargs)
class GeneticFuzzyControllerFlappyBird:
    def __init__(self, n_mfs=3):
        self.n_inputs = 7  # Updated to 7 inputs
        self.n_mfs = n_mfs
        # Define bounds for each of thse 7 inputs based on observation space
        self.mf_bounds = [
            (0, 336),       # obs[0]: horizontal distance to next pipe
            (-256, 256),    # obs[1]: vertical distance to top of next gap
            (-256, 256),    # obs[2]: vertical distance to bottom of next gap
            (0, 512),       # obs[3]: bird's vertical position (y)
            (-8, 10),      # obs[4]: bird's velocity
            (0, 336),       # obs[5]: horizontal distance to next-next pipe
            (-256, 256)     # obs[9]: vertical distance to next-next gap's top
        ]
        self.rule_bounds = (-2, 2)
        self.n_rules = n_mfs ** self.n_inputs
        self.n_rule_params = self.n_inputs + 1 #ax + by + c so its 7 + 1
        self.total_rule_params = self.n_rules * self.n_rule_params

    def triangular_mf(self, x, params):
        a, b, c = params
        if x <= a or x >= c:
            return 0.0
        if x <= b:
            return (x - a) / (b - a + 1e-6)
        else:
            return (c - x) / (c - b + 1e-6)

    def fuzzy_inference(self, state, individual):
        # Extract centers for each input
        centers = []
        start = 0
        for i in range(self.n_inputs):
            centers_i = np.sort(individual[start:start+self.n_mfs])
            centers.append(centers_i)
            start += self.n_mfs
        rule_params = individual[start:]
        
        # Build membership functions for each input
        mfs = []
        for i in range(self.n_inputs):
            lower, upper = self.mf_bounds[i]
            c = centers[i]
            mf_i = [
                (lower, c[0], c[1]),   # left shoulder
                (c[0], c[1], c[2]),    # middle
                (c[1], c[2], upper)    # right shoulder
            ]
            mfs.append(mf_i)
        
        # Compute membership values
        membership_values = []
        for i in range(self.n_inputs):
            x_val = state[i]
            mf_vals = [self.triangular_mf(x_val, mf) for mf in mfs[i]]
            membership_values.append(mf_vals)
        
        # Compute rule activations and outputs
        activations = []
        outputs = []
        rule_index = 0
        for idx in np.ndindex(*(self.n_mfs,)*self.n_inputs):
            act = min(membership_values[i][idx[i]] for i in range(self.n_inputs))
            activations.append(act)
            params = rule_params[rule_index*self.n_rule_params:(rule_index+1)*self.n_rule_params]
            output = np.dot(params[:self.n_inputs], state) + params[self.n_inputs]
            outputs.append(output)
            rule_index += 1
        
        total_activation = sum(activations) + 1e-6
        fuzzy_output = sum(act * out for act, out in zip(activations, outputs)) / total_activation
        return fuzzy_output

    def initialize_individual(self):
        centers = []
        for i in range(self.n_inputs):
            lower, upper = self.mf_bounds[i]
            centers_i = np.sort(np.random.uniform(lower, upper, self.n_mfs))
            centers.append(centers_i)
        centers = np.concatenate(centers)
        rule_params = np.random.uniform(self.rule_bounds[0], self.rule_bounds[1], self.total_rule_params)
        return np.concatenate([centers, rule_params])

    def fitness(self, individual, render=False):
        render_mode = "human" if render else None
        # Create base environment and wrap with seed control
        base_env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)
        env = SeedWrapper(base_env, seed=4255)  # Fixed seed for all evaluations
        total_reward = 0
        frames = []
        
        try:
            for episode in range(5):
                observation, _ = env.reset()
                episode_frames = []
                done = False
                
                while not done:
                    # Handle pygame events
                    if render:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                env.close()
                                return total_reward / 5

                    # Extract relevant 7 inputs from observation
                    selected_inputs = [
                        observation[0],
                        observation[1],
                        observation[2],
                        observation[3],
                        observation[4],
                        observation[5],
                        observation[9]
                    ]
                    
                    # Get fuzzy logic decision
                    fuzzy_output = self.fuzzy_inference(selected_inputs, individual)
                    action = 1 if fuzzy_output > 0 else 0
                    
                    # Take action
                    observation, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward

                    # Capture frame for GIF
                    if render:
                        frame = env.render()
                        if frame is not None:  # Some environments return None when not rendering
                            episode_frames.append(frame)

                # Add episode frames to main list
                frames.extend(episode_frames)

        finally:
            env.close()

        # Save as GIF if rendering was enabled
        if render and frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = f"flappy_bird_demo_{timestamp}.gif"
            
            # Save using imageio
            imageio.mimsave(
                gif_path,
                [np.array(frame) for frame in frames],
                fps=30,
                loop=0
            )
            print(f"Saved demo GIF to: {gif_path}")

        return total_reward / 5

    def tournament_selection(self, population, fitnesses):
        selected = []
        for _ in range(len(population)):
            candidates = np.random.choice(len(population), TOURNAMENT_SIZE, replace=False)
            best_idx = candidates[np.argmax(fitnesses[candidates])]
            selected.append(population[best_idx])
        return selected

    def crossover(self, parent1, parent2):
        child = np.empty_like(parent1)
        for i in range(self.n_inputs):
            start = i * self.n_mfs
            end = start + self.n_mfs
            split = np.random.randint(1, self.n_mfs)
            child[start:start+split] = parent1[start:start+split]
            child[start+split:end] = parent2[start+split:end]
            child[start:end] = np.sort(child[start:end])
        rule_start = self.n_inputs * self.n_mfs
        p1_rules = parent1[rule_start:]
        p2_rules = parent2[rule_start:]
        alpha = np.random.rand(len(p1_rules))
        child[rule_start:] = alpha * p1_rules + (1 - alpha) * p2_rules
        return child

    def mutate(self, individual):
        mut_ind = individual.copy()
        for i in range(self.n_inputs):
            lower, upper = self.mf_bounds[i]
            range_i = upper - lower
            start = i * self.n_mfs
            end = start + self.n_mfs
            if np.random.rand() < MUT_PROB:
                idx = np.random.randint(self.n_mfs)
                mutation_step = np.random.normal(0, 0.01 * range_i)
                mut_ind[start + idx] += mutation_step
                mut_ind[start:end] = np.sort(mut_ind[start:end])
                mut_ind[start:end] = np.clip(mut_ind[start:end], lower, upper)
        rule_start = self.n_inputs * self.n_mfs
        rule_params = mut_ind[rule_start:]
        mask = np.random.rand(len(rule_params)) < MUT_PROB
        noise = np.random.normal(0, 0.1, len(rule_params))
        rule_params = np.clip(rule_params + mask * noise, self.rule_bounds[0], self.rule_bounds[1])
        mut_ind[rule_start:] = rule_params
        return mut_ind

    def run_evolution(self):
        population = [self.initialize_individual() for _ in range(POP_SIZE)]
        best_fitness_history = []
        for gen in range(N_GENERATIONS):
            fitnesses = np.array([self.fitness(ind) for ind in population])
            best_idx = np.argmax(fitnesses)
            best_fitness_history.append(fitnesses[best_idx])
            print(f"Gen {gen+1}: Best Fitness (reward) = {best_fitness_history[-1]:.4f}")
            selected = self.tournament_selection(population, fitnesses)
            offspring = []
            for i in range(0, len(population), 2):
                p1 = selected[i]
                p2 = selected[(i+1) % len(population)]
                if np.random.rand() < CX_PROB:
                    c1 = self.crossover(p1, p2)
                    c2 = self.crossover(p2, p1)
                    offspring.extend([c1, c2])
                else:
                    offspring.extend([p1, p2])
            population = [self.mutate(ind) for ind in offspring]
        best_ind = population[np.argmax([self.fitness(ind) for ind in population])]
        demo_reward = self.fitness(best_ind, render=True)
        print(f"\nDemo run reward: {demo_reward}")
        plt.plot(best_fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (reward)')
        plt.title('Evolution Progress')
        plt.show()

if __name__ == '__main__':
    controller = GeneticFuzzyControllerFlappyBird(n_mfs=3)
    controller.run_evolution()