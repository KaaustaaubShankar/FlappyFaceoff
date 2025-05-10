import gymnasium as gym
from gym.wrappers import RecordVideo
import flappy_bird_gymnasium
import numpy as np
import matplotlib.pyplot as plt
import pygame
import os
from datetime import datetime
from itertools import product
from multiprocessing import Pool
from tqdm import tqdm

# Constants for the two phases
PHASE1_POP_SIZE = 100000
PHASE1_GENERATIONS = 20
PHASE2_POP_SIZE = 10000
PHASE2_GENERATIONS = 80
CX_PROB = 0.8
MUT_PROB = 0.3

class GeneticFuzzySystemSevenInputs:
    mf_bounds = [0, 1]
    rule_bounds = [-1, 1]

    def __init__(self):
        self.best_individual = None
        self.fitness_history = {
            'best': [],
            'average': []
        }
        self.population = []
        self.fitnesses = []

    @staticmethod
    def triangular_mf(x, params):
        a, b, c = params
        if x == b:
            return 1
        if x <= a or x >= c:
            return 0.0
        return (x - a)/(b - a) if x <= b else (c - x)/(c - b)

    @staticmethod
    def fuzzy_inference(inputs, individual):
        centers = individual[:7]
        rule_consequents = individual[7:]
        
        membership_grades = []
        for i in range(7):
            center = centers[i]
            mfs = [(0, 0, center), (0, center, 1), (center, 1, 1)]
            grades = [GeneticFuzzySystemSevenInputs.triangular_mf(inputs[i], mf) 
                     for mf in mfs]
            membership_grades.append(grades)
        
        activations = []
        for indices in product(range(3), repeat=7):
            activation = min(membership_grades[i][indices[i]] for i in range(7))
            activations.append(activation)
        
        total = sum(activations) + 1e-6
        return np.dot(activations, rule_consequents) / total

    @staticmethod
    def fitness(individual, render=False):
        if not pygame.get_init():
            pygame.init()
            
        render_mode = "rgb_array" if render else None
        env = gym.make("FlappyBird-v0", render_mode=render_mode, 
                      use_lidar=False, normalize_obs=True)
        
        if render:
            os.makedirs("videos", exist_ok=True)
            env = RecordVideo(
                env=env,
                video_folder="videos",
                name_prefix=f"flappy_best_{datetime.now().strftime('%Y%m%d%H%M%S')}",
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
                    observation[6], observation[7], observation[8],
                    observation[3], observation[4], observation[5],
                    observation[9]
                ]
                
                fuzzy_output = GeneticFuzzySystemSevenInputs.fuzzy_inference(
                    selected_inputs, individual)
                action = 1 if fuzzy_output > 0 else 0
                
                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
        finally:
            env.close()
        
        return total_reward

    def tournament_selection(self, population, fitnesses, num_selected, tournament_size):
        selected = []
        for _ in range(num_selected):
            candidates = np.random.choice(len(population), tournament_size, replace=False)
            best_idx = candidates[np.argmax(fitnesses[candidates])]
            selected.append(population[best_idx])
        return selected

    def initialize_individual(self):
        centers = np.random.uniform(*self.mf_bounds, 7)
        rules = np.random.uniform(*self.rule_bounds, 3**7)
        return np.concatenate([centers, rules])

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for i in range(7):
            if np.random.rand() < 0.5:
                child[i] = parent2[i]
        alpha = np.random.rand(3**7)
        child[7:] = alpha * parent1[7:] + (1 - alpha) * parent2[7:]
        return child

    def mutate(self, individual):
        mutated = individual.copy()
        for i in range(7):
            if np.random.rand() < MUT_PROB/7:
                mutated[i] += np.random.normal(0, 0.1)
                mutated[i] = np.clip(mutated[i], *self.mf_bounds)
        mask = np.random.rand(3**7) < MUT_PROB
        noise = np.random.normal(0, 0.1, 3**7)
        mutated[7:] = np.clip(mutated[7:] + mask * noise, *self.rule_bounds)
        return mutated

    def run_evolution_phase(self, pop_size, n_generations, initial_population=None):
        if initial_population is None:
            population = [self.initialize_individual() for _ in range(pop_size)]
        else:
            population = initial_population.copy()
            # Fill or truncate to match pop_size
            while len(population) < pop_size:
                population.append(self.initialize_individual())
            population = population[:pop_size]

        for gen in range(n_generations):
            with Pool() as pool:
                fitnesses = []
                with tqdm(total=len(population), desc=f"Gen {gen+1}") as pbar:
                    for result in pool.imap(self.fitness, population):
                        fitnesses.append(result)
                        pbar.update()
            fitnesses = np.array(fitnesses)
            
            # Update fitness history
            self.fitness_history['best'].append(np.max(fitnesses))
            self.fitness_history['average'].append(np.mean(fitnesses))
            print(f"Gen {gen+1}: Best = {self.fitness_history['best'][-1]:.1f}, "
                  f"Avg = {self.fitness_history['average'][-1]:.1f}")
            
            # Dynamic parameters based on current pop_size
            elitism_size = int(0.01 * pop_size)
            tournament_size = int(0.02 * pop_size)
            
            # Elitism
            sorted_indices = np.argsort(fitnesses)[::-1]
            elites = [population[i] for i in sorted_indices[:elitism_size]]
            
            # Generate offspring
            num_parents_needed = (pop_size - elitism_size) * 2
            selected = self.tournament_selection(population, fitnesses, num_parents_needed, tournament_size)
            
            offspring = []
            for i in range(0, num_parents_needed, 2):
                p1, p2 = selected[i], selected[i+1]
                if np.random.rand() < CX_PROB:
                    offspring.append(self.crossover(p1, p2))
                    offspring.append(self.crossover(p2, p1))
                else:
                    offspring.extend([p1, p2])
            
            # Mutation and new population
            offspring = [self.mutate(ind) for ind in offspring[:pop_size - elitism_size]]
            population = elites + offspring
        
        # Save final population and fitnesses
        self.population = population
        self.fitnesses = fitnesses
        return population

if __name__ == "__main__":
    pygame.init()
    gfs = GeneticFuzzySystemSevenInputs()
    
    # Phase 1: Large population exploration
    print("=== Phase 1: Initial Exploration ===")
    phase1_pop = gfs.run_evolution_phase(PHASE1_POP_SIZE, PHASE1_GENERATIONS)
    
    # Select top individuals for phase 2
    sorted_indices = np.argsort(gfs.fitnesses)[::-1]
    top_individuals = [phase1_pop[i] for i in sorted_indices[:PHASE2_POP_SIZE]]
    
    # Phase 2: Focused optimization
    print("\n=== Phase 2: Focused Optimization ===")
    gfs.run_evolution_phase(PHASE2_POP_SIZE, PHASE2_GENERATIONS, top_individuals)
    
    # Final evaluation and visualization
    print("\n=== Final Evaluation ===")
    plt.figure(figsize=(10, 6))
    plt.plot(gfs.fitness_history['best'], label='Best Fitness')
    plt.plot(gfs.fitness_history['average'], label='Average Fitness')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolution of Fitness Scores (Two-phase Evolution)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Get best individual from final population
    best_idx = np.argmax(gfs.fitnesses)
    gfs.best_individual = gfs.population[best_idx]
    print("\nRunning best individual with rendering...")
    final_score = gfs.fitness(gfs.best_individual, render=True)
    print(f"Final demonstration score: {final_score}")
    
    # Plot membership functions
    gfs.plot_membership_functions(gfs.best_individual)

    def plot_membership_functions(self, individual):
        plt.figure(figsize=(15, 10))
        centers = individual[:7]
        for i in range(7):
            plt.subplot(3, 3, i+1)
            x = np.linspace(0, 1, 100)
            mfs = [
                (0, 0, centers[i]),
                (0, centers[i], 1),
                (centers[i], 1, 1)
            ]
            for params in mfs:
                y = [self.triangular_mf(xi, params) for xi in x]
                plt.plot(x, y)
            plt.title(f"Input {i+1} MFs (Center: {centers[i]:.2f})")
        plt.tight_layout()
        plt.show()