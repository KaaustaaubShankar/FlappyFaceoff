import gymnasium as gym
from gym.wrappers import RecordVideo
import flappy_bird_gymnasium
import numpy as np
import matplotlib.pyplot as plt
import pygame
import os
from datetime import datetime
from itertools import product
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Constants
POP_SIZE = 1000
N_GENERATIONS = 100
CX_PROB = 0.8
MUT_PROB = 0.4
TOURNAMENT_SIZE = int(0.02 * POP_SIZE)
ELITISM_SIZE = int(0.01 * POP_SIZE)

# Multiprocessing globals
worker_env = None


def init_worker():
    """Initialize worker process with persistent environment"""
    global worker_env
    try:
        pygame.init()
        worker_env = gym.make(
            "FlappyBird-v0",
            render_mode=None,
            use_lidar=True,
            normalize_obs=True,
            disable_env_checker=True
        )
    except Exception as e:
        print(f"Worker initialization failed: {e}")


def close_worker():
    """Cleanup worker process resources"""
    global worker_env
    if worker_env is not None:
        worker_env.close()
    pygame.quit()


def _derive_fuzzy_inputs(observation):
    """
    Turn a length-180 LIDAR array into 6 scalar inputs by
    partitioning into 6 equal sectors and taking the min in each.
    Returns:
      - inputs: list of 6 floats
    """
    # Extract lidar readings as a plain Python list
    if isinstance(observation, dict):
        lidar = list(observation['lidar'])
    else:
        lidar = list(observation)
    n = len(lidar)
    sector_size = n // 6  # 180//6 = 30
    features = []
    for i in range(6):
        start = i * sector_size
        end = (i + 1) * sector_size if i < 5 else n
        sector = lidar[start:end]
        features.append(float(min(sector)))  # Python float, not numpy
    #print(f"Derived features: {features}")
    return features


class GeneticFuzzySystemSixInputs:
    N_INPUTS = 6
    mf_bounds = [0, 1]
    rule_bounds = [-1, 1]

    def __init__(self):
        self.best_individual = None
        self.fitness_history = {'best': [], 'average': []}

    @staticmethod
    def triangular_mf(x, params):
        a, b, c = params
        if x == b:
            return 1.0
        if x <= a or x >= c:
            return 0.0
        return (x - a)/(b - a) if x <= b else (c - x)/(c - b)

    @staticmethod
    def fuzzy_inference(inputs, individual):
        centers = individual[:GeneticFuzzySystemSixInputs.N_INPUTS]
        rule_consequents = individual[GeneticFuzzySystemSixInputs.N_INPUTS:]
        membership_grades = []
        for i in range(GeneticFuzzySystemSixInputs.N_INPUTS):
            center = centers[i]
            mfs = [(0, 0, center), (0, center, 1), (center, 1, 1)]
            grades = [GeneticFuzzySystemSixInputs.triangular_mf(inputs[i], mf)
                      for mf in mfs]
            membership_grades.append(grades)
        activations = []
        for indices in product(range(3), repeat=GeneticFuzzySystemSixInputs.N_INPUTS):
            activation = min(membership_grades[i][indices[i]] for i in range(GeneticFuzzySystemSixInputs.N_INPUTS))
            activations.append(activation)
        total = sum(activations) + 1e-6
        return np.dot(activations, rule_consequents) / total

    def evaluate_individual(self, individual):
        """Evaluation function for parallel processing"""
        global worker_env
        total_reward = 0.0
        last_clear = None
        try:
            observation, _ = worker_env.reset(seed=42)
            done = False
            while not done:
                inputs= _derive_fuzzy_inputs(observation)
                fuzzy_output = self.fuzzy_inference(inputs, individual)
                action = 1 if fuzzy_output > 0 else 0
                observation, reward, terminated, truncated, _ = worker_env.step(action)
                done = terminated or truncated
                total_reward += reward
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return 0.0
        return total_reward

    def fitness(self, individual, render=False):
        if render:
            env = gym.make(
                "FlappyBird-v0",
                render_mode="rgb_array",
                use_lidar=True,
                normalize_obs=True
            )
            env = RecordVideo(
                env=env,
                video_folder="videos",
                name_prefix=f"flappy_best_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                episode_trigger=lambda x: True
            )
            total_reward = 0.0
            last_clear = None
            try:
                observation, _ = env.reset(seed=42)
                done = False
                while not done:
                    inputs = _derive_fuzzy_inputs(observation)
                    fuzzy_output = self.fuzzy_inference(inputs, individual)
                    action = 1 if fuzzy_output > 0 else 0
                    observation, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
            finally:
                env.close()
            return total_reward
        else:
            raise RuntimeError("Non-render evaluation should use worker processes")

    def tournament_selection(self, population, fitnesses, num_selected):
        selected = []
        for _ in range(num_selected):
            candidates = np.random.choice(len(population), TOURNAMENT_SIZE, replace=False)
            best_idx = candidates[np.argmax(fitnesses[candidates])]
            selected.append(population[best_idx])
        return selected

    def initialize_individual(self):
        centers = np.random.uniform(*self.mf_bounds, self.N_INPUTS)
        rules = np.random.uniform(*self.rule_bounds, 3**self.N_INPUTS)
        return np.concatenate([centers, rules])

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for i in range(self.N_INPUTS):
            if np.random.rand() < 0.5:
                child[i] = parent2[i]
        alpha = np.random.rand(3**self.N_INPUTS)
        child[self.N_INPUTS:] = alpha * parent1[self.N_INPUTS:] + (1 - alpha) * parent2[self.N_INPUTS:]
        return child

    def mutate(self, individual, generation_ratio, spike=False):
        mutated = individual.copy()
        center_mut_std = 0.2 * (1 - generation_ratio)
        rule_mut_std = 0.5 * (1 - generation_ratio)
        if spike:
            center_mut_std *= 1.5
            rule_mut_std *= 1.5
        # Mutate centers
        for i in range(self.N_INPUTS):
            if np.random.rand() < MUT_PROB / self.N_INPUTS:
                mutated[i] += np.random.normal(0, center_mut_std)
                mutated[i] = np.clip(mutated[i], *self.mf_bounds)
        # Mutate rules
        mask = np.random.rand(3**self.N_INPUTS) < MUT_PROB
        noise = np.random.normal(0, rule_mut_std, 3**self.N_INPUTS)
        mutated[self.N_INPUTS:] = np.clip(mutated[self.N_INPUTS:] + mask * noise, *self.rule_bounds)
        return mutated

    def run_evolution(self):
        with Pool(
            processes=cpu_count(),
            initializer=init_worker,
            initargs=(),
        ) as pool:
            population = [self.initialize_individual() for _ in range(POP_SIZE)]
            last_best = -np.inf
            stagnation_counter = 0
            for gen in range(N_GENERATIONS):
                chunksize = max(1, POP_SIZE // (cpu_count() * 4))
                fitnesses = np.array(list(
                    tqdm(
                        pool.imap(self.evaluate_individual, population, chunksize=chunksize),
                        total=POP_SIZE,
                        desc=f"Generation {gen+1}/{N_GENERATIONS}"
                    )
                ))
                best_fitness = np.max(fitnesses)
                avg_fitness = np.mean(fitnesses)
                self.fitness_history['best'].append(best_fitness)
                self.fitness_history['average'].append(avg_fitness)
                print(f"\nGen {gen+1}: Best={best_fitness:.1f}, Avg={avg_fitness:.1f}")
                if best_fitness <= last_best:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                    last_best = best_fitness
                sorted_idx = np.argsort(fitnesses)[::-1]
                elites = [population[i] for i in sorted_idx[:ELITISM_SIZE]]
                num_parents = (POP_SIZE - ELITISM_SIZE) * 2
                selected = self.tournament_selection(population, fitnesses, num_parents)
                offspring = []
                for i in range(0, num_parents, 2):
                    p1, p2 = selected[i], selected[i+1]
                    if np.random.rand() < CX_PROB:
                        offspring.append(self.crossover(p1, p2))
                        offspring.append(self.crossover(p2, p1))
                    else:
                        offspring.extend([p1, p2])
                generation_ratio = gen / N_GENERATIONS
                offspring = [self.mutate(ind, generation_ratio, stagnation_counter >= 3)
                             for ind in offspring[:POP_SIZE - ELITISM_SIZE]]
                population = elites + offspring
        self.best_individual = elites[0]
        self.plot_results()
        self.run_final_demo()

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.fitness_history['best'], 'b-', label='Best Fitness')
        plt.plot(self.fitness_history['average'], 'g--', label='Average Fitness')
        for gen in range(0, len(self.fitness_history['best']), 5):
            plt.scatter(gen, self.fitness_history['best'][gen])
            plt.text(gen, self.fitness_history['best'][gen], f"{self.fitness_history['best'][gen]:.1f}", fontsize=8, ha='right')
        last_gen = len(self.fitness_history['best']) - 1
        plt.scatter(last_gen, self.fitness_history['best'][last_gen])
        plt.text(last_gen, self.fitness_history['best'][last_gen], f"{self.fitness_history['best'][last_gen]:.1f}", fontsize=8, ha='right')
        plt.xlabel("Generation")
        plt.ylabel("Fitness Score")
        plt.title("Evolutionary Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig("fitness_progress.png")
        plt.show()

    def run_final_demo(self):
        print("\nRunning final demonstration...")
        final_score = self.fitness(self.best_individual, render=True)
        print(f"Final Score: {final_score}")
        self.plot_membership_functions()

    def plot_membership_functions(self):
        plt.figure(figsize=(15, 10))
        centers = self.best_individual[:self.N_INPUTS]
        for i in range(self.N_INPUTS):
            plt.subplot(3, 3, i+1)
            x = np.linspace(0, 1, 100)
            mfs = [(0, 0, centers[i]), (0, centers[i], 1), (centers[i], 1, 1)]
            for params in mfs:
                y = [self.triangular_mf(xi, params) for xi in x]
                plt.plot(x, y)
            plt.title(f"Input {i+1} Membership Functions")
        plt.tight_layout()
        plt.savefig("membership_functions.png")
        plt.show()

if __name__ == "__main__":
    pygame.init()
    try:
        gfs = GeneticFuzzySystemSixInputs()
        gfs.run_evolution()
    finally:
        pygame.quit()
