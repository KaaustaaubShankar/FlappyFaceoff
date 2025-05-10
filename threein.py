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
POP_SIZE = 10000
N_GENERATIONS = 100
CX_PROB = 0.7
MUT_PROB = 0.4
TOURNAMENT_SIZE = int(0.02 * POP_SIZE)
ELITISM_SIZE = int(0.01 * POP_SIZE)

# Multiprocessing globals
worker_env = None

def init_worker():
    global worker_env
    try:
        pygame.init()
        worker_env = gym.make(
            "FlappyBird-v0",
            render_mode=None,
            use_lidar=False,
            normalize_obs=True
        )
    except Exception as e:
        print(f"Worker initialization failed: {e}")

def close_worker():
    global worker_env
    if worker_env is not None:
        worker_env.close()
    pygame.quit()

class GeneticFuzzySystemThreeInputs:
    mf_bounds = [0, 1]
    rule_bounds = [-5, 5]

    def __init__(self):
        self.best_individual = None
        self.fitness_history = {'best': [], 'average': []}

    @staticmethod
    def triangular_mf(x, params):
        a, b, c = params
        if x == b:
            return 1
        if x <= a or x >= c:
            return 0.0
        return (x - a)/(b - a) if x <= b else (c - x)/(c - b)

    def fuzzy_inference(self, inputs, individual):
        centers = individual[:3]
        rules_params = individual[3:].reshape(27, 4)  # 27 rules, 4 parameters each
        
        membership_grades = []
        for i in range(3):
            center = centers[i]
            mfs = [(0, 0, center), (0, center, 1), (center, 1, 1)]
            grades = [self.triangular_mf(inputs[i], mf) for mf in mfs]
            membership_grades.append(grades)
        
        activations = []
        rule_outputs = []
        for idx, indices in enumerate(product(range(3), repeat=3)):
            activation = 1.0
            for i in range(3):
                activation *= membership_grades[i][indices[i]]
            activations.append(activation)
            
            # TSK rule: w0 + w1*x1 + w2*x2 + w3*x3
            w0, w1, w2, w3 = rules_params[idx]
            output = w0 + w1*inputs[0] + w2*inputs[1] + w3*inputs[2]
            rule_outputs.append(output)
        
        total = sum(activations) + 1e-6
        return np.dot(activations, rule_outputs) / total

    def evaluate_individual(self, individual):
        global worker_env
        total_reward = 0
        try:
            observation, _ = worker_env.reset(seed=42)
            done = False
            
            while not done:
                selected_inputs = [observation[3], observation[4], observation[9]]
                fuzzy_output = self.fuzzy_inference(selected_inputs, individual)
                action = 1 if fuzzy_output > 0 else 0
                
                observation, reward, terminated, truncated, _ = worker_env.step(action)
                done = terminated or truncated
                total_reward += reward
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return 0
        
        return total_reward

    def fitness(self, individual, render=False):
        if render:
            env = gym.make("FlappyBird-v0", 
                          render_mode="rgb_array",
                          use_lidar=False,
                          normalize_obs=True)
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
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            return total_reward
                    
                    selected_inputs = [observation[3], observation[4], observation[9]]
                    fuzzy_output = self.fuzzy_inference(selected_inputs, individual)
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
        centers = np.random.uniform(*self.mf_bounds, 3)
        rules = np.random.uniform(*self.rule_bounds, 27*4)
        return np.concatenate([centers, rules])

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        # Crossover centers
        for i in range(3):
            if np.random.rand() < 0.5:
                child[i] = parent2[i]
        
        # Crossover rules with per-rule blending
        parent1_rules = parent1[3:].reshape(27, 4)
        parent2_rules = parent2[3:].reshape(27, 4)
        alpha = np.random.rand(27)
        child_rules = np.array([
            alpha[i] * parent1_rules[i] + (1 - alpha[i]) * parent2_rules[i]
            for i in range(27)
        ])
        child[3:] = child_rules.reshape(-1)
        return child

    def mutate(self, individual, generation_ratio, spike=False):
        mutated = individual.copy()
        center_mut_std = 0.2 * (1 - generation_ratio)
        rule_mut_std = 0.5 * (1 - generation_ratio)

        if spike:
            center_mut_std *= 1.5
            rule_mut_std *= 1.5

        # Mutate centers
        for i in range(3):
            if np.random.rand() < MUT_PROB/3:
                mutated[i] += np.random.normal(0, center_mut_std)
                mutated[i] = np.clip(mutated[i], *self.mf_bounds)

        # Mutate rules
        mask = np.random.rand(108) < MUT_PROB
        noise = np.random.normal(0, rule_mut_std, 108)
        mutated[3:] = np.clip(mutated[3:] + mask * noise, *self.rule_bounds)

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
                fitnesses = list(tqdm(
                    pool.imap(self.evaluate_individual, population, chunksize=chunksize),
                    total=POP_SIZE,
                    desc=f"Generation {gen+1}/{N_GENERATIONS}"
                ))
                fitnesses = np.array(fitnesses)

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

                sorted_indices = np.argsort(fitnesses)[::-1]
                elites = [population[i] for i in sorted_indices[:ELITISM_SIZE]]

                num_parents_needed = (POP_SIZE - ELITISM_SIZE) * 2
                selected = self.tournament_selection(population, fitnesses, num_parents_needed)

                offspring = []
                for i in range(0, num_parents_needed, 2):
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
            plt.scatter(gen, self.fitness_history['best'][gen], color='blue')
            plt.text(gen, self.fitness_history['best'][gen], f"{self.fitness_history['best'][gen]:.1f}", 
                     fontsize=8, color='blue', ha='right')
        
        last_gen = len(self.fitness_history['best']) - 1
        plt.scatter(last_gen, self.fitness_history['best'][last_gen], color='blue')
        plt.text(last_gen, self.fitness_history['best'][last_gen], f"{self.fitness_history['best'][last_gen]:.1f}", 
                 fontsize=8, color='blue', ha='right')
        
        plt.xlabel("Generation")
        plt.ylabel("Fitness Score")
        plt.title("Evolutionary Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig("threein_fitness_progress.png")
        plt.show()


    def run_final_demo(self):
        print("\nRunning final demonstration...")
        final_score = self.fitness(self.best_individual, render=True)
        print(f"Final Score: {final_score}")
        self.plot_membership_functions()

    def plot_membership_functions(self):
        plt.figure(figsize=(15, 5))
        centers = self.best_individual[:3]
        for i in range(3):
            plt.subplot(1, 3, i+1)
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
        gfs = GeneticFuzzySystemThreeInputs()
        gfs.run_evolution()
    finally:
        pygame.quit()