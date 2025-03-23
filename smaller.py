import gymnasium as gym
from gym.wrappers import RecordVideo
import flappy_bird_gymnasium
import numpy as np
import matplotlib.pyplot as plt
import pygame
import imageio
import os
from datetime import datetime

pygame.init()

# Global GA parameters
POP_SIZE = 1000
N_GENERATIONS = 50
TOURNAMENT_SIZE = 3
CX_PROB = 0.7
MUT_PROB = 0.2


class GeneticFuzzyControllerFlappyBird:
    def __init__(self, n_mfs=5):
        self.n_inputs = 3 
        self.n_mfs = n_mfs
        # Define bounds for each of these 3 inputs based on observation space
        self.mf_bounds = [
            (0, 1),       # obs[0]: horizontal distance to next pipe
            (-1, 1),    # obs[2]: vertical distance to bottom of next gap
            (-8, 10),      # obs[4]: bird's velocity
        ]
        self.rule_bounds = (-2, 2)
        self.n_rules = n_mfs ** self.n_inputs
        self.n_rule_params = self.n_inputs + 1  # ax + by + cz + d
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
        #print(state)
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
            mf_i = []
            for j in range(self.n_mfs):
                if j == 0:
                    a = lower
                    b = c[j]
                    next_c = c[j+1] if (j+1) < len(c) else upper
                elif j == self.n_mfs - 1:
                    a = c[j-1]
                    b = c[j]
                    next_c = upper
                else:
                    a = c[j-1]
                    b = c[j]
                    next_c = c[j+1]
                mf_i.append((a, b, next_c))
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
            activation = min(membership_values[dim][mf_idx] for dim, mf_idx in enumerate(idx))
            activations.append(activation)
            
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
        render_mode = "rgb_array" if render else None
        env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)
        
        if render:
            os.makedirs("videos", exist_ok=True)
            env = RecordVideo(
                env=env,
                video_folder="videos",
                name_prefix=f"flappy_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                episode_trigger=lambda x: True
            )
        
        total_reward = 0
        try:
            observation, _ = env.reset(seed=42)
            done = False
            
            while not done:
                if render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            return total_reward

                selected_inputs = [
                    observation[0],  # horizontal distance
                    observation[2],  # vertical distance
                    observation[4],  # velocity
                ]
                
                fuzzy_output = self.fuzzy_inference(selected_inputs, individual)
                action = 1 if fuzzy_output > 0 else 0
                
                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

        finally:
            env.close()
        
        return total_reward

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
            fitnesses = np.array([self.fitness(ind,render=False) for ind in population])
            best_idx = np.argmax(fitnesses)
            best_fitness_history.append(fitnesses[best_idx])
            print(f"Gen {gen+1}: Best Fitness (reward) = {best_fitness_history[-1]:.4f}")
            
            # Elitism
            elite = population[best_idx]
            
            selected = self.tournament_selection(population, fitnesses)
            offspring = []
            for i in range(0, len(population) - 1, 2):
                p1 = selected[i]
                p2 = selected[(i+1) % len(selected)]
                if np.random.rand() < CX_PROB:
                    c1 = self.crossover(p1, p2)
                    c2 = self.crossover(p2, p1)
                    offspring.extend([c1, c2])
                else:
                    offspring.extend([p1, p2])
            
            offspring = [self.mutate(ind) for ind in offspring]
            population = offspring + [elite]
        
        best_ind = population[np.argmax([self.fitness(ind) for ind in population])]
        demo_reward = self.fitness(best_ind, render=True)
        print(f"\nDemo run reward: {demo_reward}")
        plt.plot(best_fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (reward)')
        plt.title('Evolution Progress')
        plt.show()


if __name__ == '__main__':
    controller = GeneticFuzzyControllerFlappyBird(n_mfs=5)  # Now supports more than 3 MFs
    controller.run_evolution()