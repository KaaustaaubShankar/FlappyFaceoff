import gymnasium as gym
from gym.wrappers import RecordVideo
import flappy_bird_gymnasium
import numpy as np
import matplotlib.pyplot as plt
import pygame
import os
import time
from datetime import datetime

pygame.init()

# Global GA parameters
POP_SIZE = 1000
N_GENERATIONS = 50
TOURNAMENT_SIZE = 20
CX_PROB = 0.7
MUT_PROB = 0.2
N_RULES = 318  # Number of random rules


class GeneticFuzzyControllerFlappyBird:
    def __init__(self):
        self.n_inputs = 7
        self.n_mfs = 3  # Fixed to 3 MFs per input
        self.n_rules = N_RULES
        self.mf_bounds = [(0, 1)] * self.n_inputs
        self.rule_bounds = (-1, 1)

        # Individual component sizes
        self.centers_size = self.n_inputs  # One center per input
        self.antecedents_size = self.n_rules * self.n_inputs
        self.consequents_size = self.n_rules

    def triangular_mf(self, x, params):
        a, b, c = params
        if a == b == c:
            return 1.0 if x == a else 0.0
        if a == b:  # Left trapezoid/triangle
            if x <= a:
                return 1.0
            elif a < x <= c:
                return (c - x) / (c - a + 1e-6)
            else:
                return 0.0
        elif b == c:  # Right trapezoid/triangle
            if x >= b:
                return 1.0
            elif a <= x < b:
                return (x - a) / (b - a + 1e-6)
            else:
                return 0.0
        else:  # Normal triangle
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a + 1e-6)
            else:
                return (c - x) / (c - b + 1e-6)

    def fuzzy_inference(self, state, individual):
        # Split individual into components
        centers = individual[:self.centers_size]
        antecedents = individual[self.centers_size:self.centers_size+self.antecedents_size].astype(int)
        antecedents = np.clip(antecedents, 0, self.n_mfs-1).reshape(self.n_rules, self.n_inputs)
        consequents = individual[-self.consequents_size:]

        # Build membership functions
        mfs = []
        for i in range(self.n_inputs):
            lower, upper = self.mf_bounds[i]
            c = centers[i]
            mfs.append([
                (lower, lower, c),    # Left MF
                (lower, c, upper),    # Middle MF
                (c, upper, upper)     # Right MF
            ])

        # Calculate rule activations
        activations = []
        for rule_idx in range(self.n_rules):
            activation = 1.0
            for input_idx in range(self.n_inputs):
                mf_idx = antecedents[rule_idx, input_idx]
                membership = self.triangular_mf(state[input_idx], mfs[input_idx][mf_idx])
                activation = min(activation, membership)
            activations.append(activation)

        # Defuzzify
        total_activation = sum(activations) + 1e-6
        return np.dot(activations, consequents) / total_activation

    def initialize_individual(self):
        centers = np.array([np.random.uniform(low, high) for (low, high) in self.mf_bounds])
        antecedents = np.random.randint(0, self.n_mfs, self.antecedents_size)
        consequents = np.random.uniform(*self.rule_bounds, self.consequents_size)
        return np.concatenate([centers, antecedents, consequents])

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
                    observation[0],  # Horizontal to next pipe
                    observation[1],  # Vertical to top gap
                    observation[2],  # Vertical to bottom gap
                    observation[3],  # Bird's Y position
                    observation[4],  # Bird's velocity
                    observation[5],  # Horizontal to next-next pipe
                    observation[9]    # Vertical to next-next top
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
        
        # Mutate centers
        for i in range(self.n_inputs):
            if np.random.rand() < MUT_PROB:
                mutant[i] += np.random.normal(0, 0.05)
                mutant[i] = np.clip(mutant[i], 0, 1)
        
        # Mutate antecedents
        ante_start = self.centers_size
        ante_end = ante_start + self.antecedents_size
        mask = np.random.rand(self.antecedents_size) < MUT_PROB
        mutant[ante_start:ante_end][mask] = np.random.randint(0, self.n_mfs, np.sum(mask))
        
        # Mutate consequents
        mask = np.random.rand(self.consequents_size) < MUT_PROB
        noise = np.random.normal(0, 0.1, np.sum(mask))
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
            
            # Selection
            selected = self.tournament_selection(population, fitnesses)
            
            # Crossover
            offspring = []
            for i in range(0, len(selected), 2):
                p1, p2 = selected[i], selected[i+1]
                if np.random.rand() < CX_PROB and i+1 < len(selected):
                    c1, c2 = self.one_point_crossover(p1, p2)
                    offspring += [c1, c2]
                else:
                    offspring += [p1.copy(), p2.copy()]
            
            # Mutation
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