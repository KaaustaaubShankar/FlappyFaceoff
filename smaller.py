import gymnasium as gym
from gym.wrappers import RecordVideo
import flappy_bird_gymnasium
import numpy as np
import matplotlib.pyplot as plt
import pygame
import os
from datetime import datetime
from itertools import product
import random

from deap import base, creator, tools, algorithms

pygame.init()

# Constants
POP_SIZE = 400
N_GENERATIONS = 50
CX_PROB = 0.8
MUT_PROB = 0.4
TOURNAMENT_SIZE = 20

# Crossover and mutation parameters for continuous variables
ALPHA_BLX = 0.5   # for centers
ETA_C = 20        # distribution index for SBX (rules)
ETA_M = 20        # distribution index for polynomial mutation

class GeneticFuzzySystemSevenInputs:
    def __init__(self):
        self.mf_bounds = [0, 1]   # centers in [0, 1]
        self.rule_bounds = [-1, 1]  # rules in [-1, 1]
        
    def triangular_mf(self, x, params):
        a, b, c = params
        if x == b:
            return 1
        if x <= a or x >= c:
            return 0.0
        return (x - a) / (b - a) if x <= b else (c - x) / (c - b)

    def fuzzy_inference(self, inputs, individual):
        centers = individual[:7]
        rule_consequents = individual[7:]
        
        membership_grades = []
        for i in range(7):
            center = centers[i]
            mfs = [
                (0, 0, center),   # left shoulder
                (0, center, 1),   # triangle centered at center
                (center, 1, 1)    # right shoulder
            ]
            xi = inputs[i]
            grades = [self.triangular_mf(xi, mf) for mf in mfs]
            membership_grades.append(grades)
        
        activations = []
        for indices in product(range(3), repeat=7):
            activation = min(membership_grades[i][indices[i]] for i in range(7))
            activations.append(activation)
        
        total = sum(activations) + 1e-6
        return np.dot(activations, rule_consequents) / total

    def fitness(self, individual, render=False):
        render_mode = "rgb_array" if render else None
        env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False, normalize_obs=True)
        
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
            for seed in [42, 43, 44, 45, 46]:
                observation, _ = env.reset(seed=seed)
                done = False
                
                while not done:
                    if render:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                env.close()
                                return total_reward

                    # Extract the 7 inputs needed for the fuzzy logic
                    selected_inputs = [
                        observation[0],  # horizontal distance to next pipe
                        observation[1],  # vertical distance to top of next gap
                        observation[2],  # vertical distance to bottom of next gap
                        observation[3],  # bird's vertical position (y)
                        observation[4],  # bird's velocity
                        observation[5],  # horizontal distance to next-next pipe
                        observation[9]   # vertical distance to next-next gap's top
                    ]
                    
                    fuzzy_output = self.fuzzy_inference(selected_inputs, individual)
                    action = 1 if fuzzy_output > 0 else 0
                    
                    observation, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward

        finally:
            env.close()
        
        return total_reward / 5  # Return average reward across seeds

    # The evaluation function for DEAP (must return a tuple)
    def evaluate(self, individual):
        return self.fitness(individual, render=False),

    # --- New advanced crossover ---
    def crossover_deap(self, ind1, ind2):
        # For the 7 center genes, use BLX-alpha blend crossover.
        for i in range(7):
            x1 = ind1[i]
            x2 = ind2[i]
            c_min, c_max = min(x1, x2), max(x1, x2)
            I = c_max - c_min
            lower_bound = c_min - ALPHA_BLX * I
            upper_bound = c_max + ALPHA_BLX * I
            # Sample new genes uniformly from the extended range.
            ind1[i] = np.clip(random.uniform(lower_bound, upper_bound), *self.mf_bounds)
            ind2[i] = np.clip(random.uniform(lower_bound, upper_bound), *self.mf_bounds)
        
        # For the 2187 rule genes, use simulated binary crossover (SBX).
        for i in range(7, len(ind1)):
            if random.random() <= 0.5:
                if abs(ind1[i] - ind2[i]) > 1e-14:
                    x1 = min(ind1[i], ind2[i])
                    x2 = max(ind1[i], ind2[i])
                    lower, upper = self.rule_bounds
                    rand = random.random()
                    beta = 1.0 + (2.0*(x1 - lower)/(x2 - x1))
                    alpha = 2.0 - beta**-(ETA_C + 1)
                    if rand <= 1.0/alpha:
                        betaq = (rand * alpha)**(1.0/(ETA_C+1))
                    else:
                        betaq = (1.0/(2.0 - rand * alpha))**(1.0/(ETA_C+1))
                    c1 = 0.5*((x1 + x2) - betaq*(x2 - x1))
                    beta = 1.0 + (2.0*(upper - x2)/(x2 - x1))
                    alpha = 2.0 - beta**-(ETA_C + 1)
                    if rand <= 1.0/alpha:
                        betaq = (rand * alpha)**(1.0/(ETA_C+1))
                    else:
                        betaq = (1.0/(2.0 - rand * alpha))**(1.0/(ETA_C+1))
                    c2 = 0.5*((x1 + x2) + betaq*(x2 - x1))
                    # Randomly assign the children to parents.
                    if random.random() < 0.5:
                        ind1[i] = np.clip(c1, lower, upper)
                        ind2[i] = np.clip(c2, lower, upper)
                    else:
                        ind1[i] = np.clip(c2, lower, upper)
                        ind2[i] = np.clip(c1, lower, upper)
        return ind1, ind2

    # --- New advanced mutation using polynomial mutation ---
    def mutate_deap(self, individual):
        # Mutate centers (first 7 genes)
        for i in range(7):
            if random.random() < MUT_PROB / 7:
                lower, upper = self.mf_bounds
                x = individual[i]
                delta1 = (x - lower) / (upper - lower)
                delta2 = (upper - x) / (upper - lower)
                rand = random.random()
                mut_pow = 1.0 / (ETA_M + 1.0)
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (ETA_M + 1))
                    deltaq = (val ** mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (ETA_M + 1))
                    deltaq = 1.0 - (val ** mut_pow)
                x = x + deltaq * (upper - lower)
                individual[i] = np.clip(x, lower, upper)
        
        # Mutate rule consequents (genes 7 to end)
        for i in range(7, len(individual)):
            if random.random() < MUT_PROB:
                lower, upper = self.rule_bounds
                x = individual[i]
                delta1 = (x - lower) / (upper - lower)
                delta2 = (upper - x) / (upper - lower)
                rand = random.random()
                mut_pow = 1.0 / (ETA_M + 1.0)
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (ETA_M + 1))
                    deltaq = (val ** mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (ETA_M + 1))
                    deltaq = 1.0 - (val ** mut_pow)
                x = x + deltaq * (upper - lower)
                individual[i] = np.clip(x, lower, upper)
        return (individual,)

    def plot_membership_functions(self, individual):
        plt.figure(figsize=(15, 10))
        centers = individual[:7]
        for i in range(7):
            plt.subplot(3, 3, i+1)
            center = centers[i]
            x_vals = np.linspace(*self.mf_bounds, 200)
            mfs = [
                (0, 0, center),
                (0, center, 1),
                (center, 1, 1)
            ]
            for mf in mfs:
                plt.plot(x_vals, [self.triangular_mf(x, mf) for x in x_vals])
            plt.title(f"Input {i+1} Membership Functions\nCenter: {center:.2f}")
        plt.tight_layout()
        plt.show()

    def run_evolution_deap(self):
        # Define DEAP types
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Register attribute generators for centers and rules.
        toolbox.register("init_center", random.random)  # uniform in [0, 1]
        toolbox.register("init_rule", lambda: random.uniform(self.rule_bounds[0], self.rule_bounds[1]))
        
        def init_individual():
            centers = [random.uniform(self.mf_bounds[0], self.mf_bounds[1]) for _ in range(7)]
            rules = [random.uniform(self.rule_bounds[0], self.rule_bounds[1]) for _ in range(2187)]
            genome = np.array(centers + rules)
            return creator.Individual(genome)
        
        toolbox.register("individual", init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register evolutionary operators
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", self.crossover_deap)
        toolbox.register("mutate", self.mutate_deap)
        toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
        
        # Create initial population
        population = toolbox.population(n=POP_SIZE)
        
        # Setup statistics to track fitness
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        logbook = tools.Logbook()
        logbook.header = ["gen", "evals", "min", "avg", "max"]
        
        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        record = stats.compile(population)
        logbook.record(gen=0, evals=len(population), **record)
        print(logbook.stream)
        
        # Begin evolution
        for gen in range(1, N_GENERATIONS + 1):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for i in range(0, len(offspring), 2):
                if random.random() < CX_PROB:
                    toolbox.mate(offspring[i], offspring[i+1])
                    del offspring[i].fitness.values
                    del offspring[i+1].fitness.values
            for mutant in offspring:
                if random.random() < MUT_PROB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            population[:] = offspring
            
            # Gather statistics for this generation
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)
        
        # Select the best individual
        best_ind = tools.selBest(population, 1)[0]
        self.best = best_ind
        print("Best individual fitness:", best_ind.fitness.values[0])
        
        # Demonstrate the best individual with rendering
        demo_reward = self.fitness(best_ind, render=True)
        print("Demo Reward:", demo_reward)
        
        self.plot_membership_functions(best_ind)


if __name__ == "__main__":
    gfs = GeneticFuzzySystemSevenInputs()
    gfs.run_evolution_deap()
