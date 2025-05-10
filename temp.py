import numpy as np
from copy import deepcopy
import gymnasium as gym


class FuzzyMF:
    def __init__(self, centers):
        self.centers = np.clip(centers, 0, 1)
        
    def triangular(self, x, a, b, c):
        return max(min((x - a)/(b - a) if b != a else 1.0, 
                      (c - x)/(c - b) if c != b else 1.0), 0)
    
    def get_membership(self, inputs):
        memberships = []
        for i in range(7):
            x = inputs[i]
            c = self.centers[i]
            
            left = self.triangular(x, 0, 0, c)
            mid = self.triangular(x, 0, c, 1)
            right = self.triangular(x, c, 1, 1)
            
            memberships.append([left, mid, right])
        return np.array(memberships)

class GeneticFuzzy:
    def __init__(self, pop_size=50, mutation_rate=0.1, crossover_prob=0.7):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_prob = crossover_prob
        
        # Each rule is: [input1_mf, input2_mf, ..., input8_mf, action]
        self.rule_length = 7 + 1  # 8 MF selections + 1 action
        self.num_rules = 2000       # Number of rules in the rule base

    def evaluate(self, individual, render_mode=None):
        self.env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False, normalize_obs=True)
        
        total_reward = 0
        
        try:
            observation, _ = env.reset(seed=42)
            done = False
            
            while not done:
                if render:
                    # Handle pygame quit event
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            return total_reward

                # Extract relevant inputs from observation
                selected_inputs = [
                    observation[0],  # horizontal distance to next pipe
                    observation[1],  # vertical distance to top of next gap
                    observation[2],  # vertical distance to bottom of next gap
                    observation[3],  # bird's vertical position (y)
                    observation[4],  # bird's velocity
                    observation[5],  # horizontal distance to next-next pipe
                    observation[9]   # vertical distance to next-next gap's top
                ]
                
                # Get fuzzy logic decision
                fuzzy_output = self.fuzzy_inference(selected_inputs, individual)
                action = 1 if fuzzy_output > 0 else 0
                
                # Take action
                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

        finally:
            # Properly close the environment
            env.close()
        
        return total_reward
        self.env = gym.make("LunarLander-v3", render_mode=render_mode)
        
        centers = individual[:7]
        rules = individual[7:].reshape(self.num_rules, self.rule_length)
        
        fuzzy = FuzzyMF(centers)
        total_reward = 0
        observation, _ = self.env.reset(seed=42)
        terminated = truncated = False
        
        while not (terminated or truncated):
            memberships = fuzzy.get_membership(observation)
            rule_outputs = []
            
            # Evaluate all rules
            for rule in rules:
                # Get MF indices (0-2) for each input
                mf_indices = np.clip(rule[:7].astype(int), 0, 2)
                
                # Calculate rule strength using min operator
                strength = 1.0
                for i in range(7):
                    strength = min(strength, memberships[i, mf_indices[i]])
                
                # Store rule output (action with strength)
                action = np.clip(int(rule[-1]), 0, 3)
                rule_outputs.append((strength, action))
            
            # Weighted average calculation for action selection
            total_strength = np.sum([strength for strength, _ in rule_outputs])  # Total strength of all rules
            if total_strength == 0:  # If no strength, select a random action
                break
            else:
                # Compute the weighted sum of actions
                weighted_action_sum = np.sum([strength * action for strength, action in rule_outputs])
                
                # Calculate the weighted average
                action = int(np.round(weighted_action_sum / total_strength))  # Round to the nearest action
            
            # Take action and accumulate reward
            observation, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
        
        self.env.close()
        return total_reward



    def tournament_selection(self, pop, fitness, k=3):
        selected = np.random.choice(len(pop), k)
        best_idx = selected[np.argmax([fitness[i] for i in selected])]
        return deepcopy(pop[best_idx])

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            point = np.random.randint(len(parent1))
            return np.concatenate([parent1[:point], parent2[point:]])
        else:
            return parent1

    def mutate(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                if i < 7:  # Mutation for centers
                    individual[i] += np.random.normal(0, 0.1)
                    individual[i] = np.clip(individual[i], -1, 1)
                else:       # Mutation for rules
                    individual[i] += np.random.normal(0, 0.5)
                    if i % self.rule_length == self.rule_length - 1:  # Action part
                        individual[i] = np.clip(individual[i], 0, 3.999)
                    else:  # MF selection part
                        individual[i] = np.clip(individual[i], 0, 2.999)
        return individual

    def run(self, generations=50):
        # Initialize population: 8 centers + (num_rules * (8 MF selections + 1 action))
        genome_size = 7 + self.num_rules * self.rule_length
        pop = [np.concatenate([
            np.random.uniform(0, 1, 7),
            np.random.uniform(0, 3, self.num_rules * self.rule_length)
        ]) for _ in range(self.pop_size)]
        
        best_fitness = -np.inf
        best_individual = None
        
        for gen in range(generations):
            fitness = [self.evaluate(ind) for ind in pop]
            
            new_pop = []
            for _ in range(self.pop_size):
                parent1 = self.tournament_selection(pop, fitness)
                parent2 = self.tournament_selection(pop, fitness)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_pop.append(child)
                
            pop = new_pop
            current_best = np.max(fitness)
            if current_best > best_fitness:
                best_fitness = current_best
                best_individual = deepcopy(pop[np.argmax(fitness)])
            
            print(f"Generation {gen+1}, Best Fitness: {best_fitness:.2f}")
            
        return best_individual, best_fitness

if __name__ == "__main__":
    gf = GeneticFuzzy(pop_size=1000, mutation_rate=0.2, crossover_prob=0.7)
    best_ind, best_fit = gf.run(generations=15)
    print(f"Best fitness: {best_fit}")
    gf.evaluate(best_ind,render_mode="human")