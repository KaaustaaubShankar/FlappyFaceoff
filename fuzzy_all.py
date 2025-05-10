import gymnasium as gym
from gym.wrappers import RecordVideo
import flappy_bird_gymnasium
import numpy as np
import matplotlib.pyplot as plt
import pygame
import os
import time
from datetime import datetime
from itertools import product

pygame.init()

# Global GA parameters
POP_SIZE = 400
N_GENERATIONS = 50
TOURNAMENT_SIZE = 20
CX_PROB = 0.7
MUT_PROB = 0.2
N_RULES = 3**7  # 2187 rules for 7 inputs with 3 MFs each

class GeneticFuzzyControllerFlappyBird:
    def __init__(self):
        self.n_inputs = 7
        self.n_mfs = 3
        self.n_rules = N_RULES
        self.mf_bounds = [(-1, 1)] * self.n_inputs #thales did it so that it would be [0,1] (default) but the environment says -1,1 so lets try both i guess
        self.rule_bounds = (-1, 1)

        # Generate all possible rule antecedents (3^7 combinations)
        self.antecedents = np.array(list(product(range(3), repeat=7)))
        
        # Individual component sizes
        self.centers_size = self.n_inputs
        self.consequents_size = self.n_rules

    def triangular_mf(self, x, params):
        a, b, c = params
        if a == b == c:
            return 1.0 if x == a else 0.0
        if a == b:  # Left trapezoid
            return 1.0 if x <= a else max(0.0, (c - x)/(c - a + 1e-6))
        elif b == c:  # Right trapezoid
            return 1.0 if x >= c else max(0.0, (x - a)/(c - a + 1e-6))
        else:  # Normal triangle
            return max(0.0, min((x - a)/(b - a + 1e-6), (c - x)/(c - b + 1e-6)))

    def fuzzy_inference(self, state, individual):
        # Split individual into components
        centers = individual[:self.centers_size]
        consequents = individual[-self.consequents_size:]

        # Build membership functions for each input
        mfs = []
        for i in range(self.n_inputs):
            lower, upper = self.mf_bounds[i]
            c = centers[i]
            mfs.append([
                (lower, lower, c),    # Left MF (0)
                (lower, c, upper),    # Middle MF (1)
                (c, upper, upper)     # Right MF (2)
            ])

        # Calculate rule activations
        activations = []
        for rule_idx in range(self.n_rules):
            activation = 1.0
            for input_idx in range(self.n_inputs):
                mf_idx = self.antecedents[rule_idx, input_idx]
                membership = self.triangular_mf(state[input_idx], mfs[input_idx][mf_idx])
                activation = min(activation, membership)
            activations.append(activation)

        # Defuzzify
        total_activation = sum(activations) + 1e-6
        return np.dot(activations, consequents) / total_activation

    def initialize_individual(self):
        centers = np.array([np.random.uniform(low, high) for (low, high) in self.mf_bounds])
        consequents = np.random.uniform(*self.rule_bounds, self.consequents_size)
        return np.concatenate([centers, consequents])

    def fitness(self, individual, render=False):
        env = gym.make("FlappyBird-v0", 
                      render_mode="rgb_array" if render else None,
                      use_lidar=False,
                      normalize_obs=True)
        
        if render:
            env = RecordVideo(env, "videos", 
                             name_prefix=f"flappy_{datetime.now().strftime('%Y%m%d%H%M%S')}")

        total_reward = 0
        observation, _ = env.reset(seed=42)
        
        try:
            while True:
                if render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            return total_reward

                # State features
                state = [
                    observation[3],
                    observation[9] - (observation[4] + observation[5])/2
                ]

                output = self.fuzzy_inference(state, individual)
                action = 1 if output > 0 else 0

                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
        finally:
            env.close()
            
        return total_reward

    def tournament_selection(self, population, fitnesses):
        selected = []
        for _ in range(len(population)):
            candidates = np.random.choice(len(population), TOURNAMENT_SIZE)
            selected.append(population[np.argmax(fitnesses[candidates])])
        return selected

    def one_point_crossover(self, parent1, parent2):
        point = np.random.randint(1, len(parent1)-1)
        return (
            np.concatenate([parent1[:point], parent2[point:]]),
            np.concatenate([parent2[:point], parent1[point:]])
        )

    def mutate(self, individual):
        mutant = individual.copy()
        
        # Mutate centers (7 values)
        for i in range(self.centers_size):
            if np.random.rand() < MUT_PROB:
                mutant[i] += np.random.normal(0, 0.1)
                mutant[i] = np.clip(mutant[i], 0, 1)
        
        # Mutate consequents (2187 values)
        mask = np.random.rand(self.consequents_size) < MUT_PROB
        noise = np.random.normal(0, 0.2, np.sum(mask))
        mutant[-self.consequents_size:][mask] += noise
        mutant[-self.consequents_size:] = np.clip(mutant[-self.consequents_size:], -1, 1)
        
        return mutant

    def run_evolution(self):
        population = [self.initialize_individual() for _ in range(POP_SIZE)]
        best_fitness = []
        
        for gen in range(N_GENERATIONS):
            fitnesses = np.array([self.fitness(ind) for ind in population])
            best_idx = np.argmax(fitnesses)
            best_fitness.append(fitnesses[best_idx])
            print(f"Gen {gen+1}: Best Fitness = {best_fitness[-1]:.1f}")
            
            # Evolutionary operations
            selected = self.tournament_selection(population, fitnesses)
            offspring = []
            
            for i in range(0, len(selected), 2):
                p1, p2 = selected[i], selected[i+1]
                if np.random.rand() < CX_PROB and i+1 < len(selected):
                    c1, c2 = self.one_point_crossover(p1, p2)
                    offspring += [c1, c2]
                else:
                    offspring += [p1.copy(), p2.copy()]
            
            population = [self.mutate(ind) for ind in offspring]
        
        # Final evaluation
        best_ind = population[np.argmax([self.fitness(ind) for ind in population])]
        print(f"Demo reward: {self.fitness(best_ind, render=True)}")
        
        plt.plot(best_fitness)
        plt.title("Evolutionary Progress")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.show()

if __name__ == "__main__":
    controller = GeneticFuzzyControllerFlappyBird()
    controller.run_evolution()