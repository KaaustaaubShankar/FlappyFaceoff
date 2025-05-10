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
            use_lidar=False,
            normalize_obs=True
        )
    except Exception as e:
        print(f"Worker initialization failed: {e}")

def close_worker():
    """Cleanup worker process resources"""
    global worker_env
    if worker_env is not None:
        worker_env.close()
    pygame.quit()

def clip01(x):
    return max(0.0, min(1.0, x))

def norm_from_neg1_pos1(x):
    return clip01((x + 1.0) * 0.5)

def _derive_fuzzy_inputs(obs, last_clear):
    (
        last_pipe_x, last_top_y, last_bot_y,
        next_pipe_x, next_top_y, next_bot_y,
        next2_pipe_x, next2_top_y, next2_bot_y,
        player_y, player_vy, player_rot
    ) = obs

    t_gap = clip01(next_pipe_x)
    gap_center = (next_top_y + next_bot_y) * 0.5
    err_now_raw = player_y - gap_center
    err_now = norm_from_neg1_pos1(err_now_raw)
    gap_size = max(1e-6, next_top_y - next_bot_y)
    clear_raw = min(player_y - next_bot_y, next_top_y - player_y)
    clear = clip01(clear_raw / gap_size)
    err_pred_raw = err_now_raw + player_vy * t_gap
    err_pred = norm_from_neg1_pos1(err_pred_raw)
    delta_clear_raw = clear - last_clear if last_clear is not None else 0.0
    delta_clear = norm_from_neg1_pos1(delta_clear_raw)
    spd_desc = clip01(1.0 - player_vy)
    upr = clip01(1.0 - abs(player_rot - 0.5) * 2.0)

    return [t_gap, err_now, err_pred, clear, delta_clear, spd_desc, upr], clear

class GeneticFuzzySystemSevenInputs:
    mf_bounds = [0, 1]
    rule_bounds = [-1, 1]

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

    def evaluate_individual(self, individual):
        """Evaluation function for parallel processing"""
        global worker_env
        total_reward = 0
        last_clear = None
        
        try:
            observation, _ = worker_env.reset(seed=42)
            done = False
            
            while not done:
                selected_inputs, last_clear = _derive_fuzzy_inputs(observation, last_clear)
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
        """Render-capable fitness function for final evaluation"""
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
            last_clear = None
            try:
                observation, _ = env.reset(seed=42)
                done = False
                
                while not done:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            return total_reward
                    
                    selected_inputs, last_clear = _derive_fuzzy_inputs(observation, last_clear)
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

    def mutate(self, individual, generation_ratio, spike=False):
        mutated = individual.copy()
        center_mut_std = 0.2 * (1 - generation_ratio)
        rule_mut_std = 0.5 * (1 - generation_ratio)

        if spike:
            center_mut_std *= 1.5
            rule_mut_std *= 1.5

        for i in range(7):
            if np.random.rand() < MUT_PROB / 7:
                mutated[i] += np.random.normal(0, center_mut_std)
                mutated[i] = np.clip(mutated[i], *self.mf_bounds)

        mask = np.random.rand(3**7) < MUT_PROB
        noise = np.random.normal(0, rule_mut_std, 3**7)
        mutated[7:] = np.clip(mutated[7:] + mask * noise, *self.rule_bounds)

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
                # Parallel evaluation
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

                # Stagnation detection
                if best_fitness <= last_best:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                    last_best = best_fitness

                # Elitism
                sorted_indices = np.argsort(fitnesses)[::-1]
                elites = [population[i] for i in sorted_indices[:ELITISM_SIZE]]

                # Selection and crossover
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

                # Mutation
                generation_ratio = gen / N_GENERATIONS
                offspring = [self.mutate(ind, generation_ratio, stagnation_counter >= 3) 
                            for ind in offspring[:POP_SIZE - ELITISM_SIZE]]

                population = elites + offspring

        # Final processing
        self.best_individual = elites[0]
        self.plot_results()
        self.run_final_demo()

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.fitness_history['best'], 'b-', label='Best Fitness')
        plt.plot(self.fitness_history['average'], 'g--', label='Average Fitness')
        
        # Add dots and text every 5 generations
        for gen in range(0, len(self.fitness_history['best']), 5):
            plt.scatter(gen, self.fitness_history['best'][gen], color='blue')
            plt.text(gen, self.fitness_history['best'][gen], f"{self.fitness_history['best'][gen]:.1f}", 
                     fontsize=8, color='blue', ha='right')
        
        # Add dot and text for the last generation
        last_gen = len(self.fitness_history['best']) - 1
        plt.scatter(last_gen, self.fitness_history['best'][last_gen], color='blue')
        plt.text(last_gen, self.fitness_history['best'][last_gen], f"{self.fitness_history['best'][last_gen]:.1f}", 
                 fontsize=8, color='blue', ha='right')
        
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
        centers = self.best_individual[:7]
        for i in range(7):
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
        gfs = GeneticFuzzySystemSevenInputs()
        gfs.run_evolution()
    finally:
        pygame.quit()