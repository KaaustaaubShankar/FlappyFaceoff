import pygame
import random
import numpy as np
import gym
from gym import spaces

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
BIRD_WIDTH = 40
BIRD_HEIGHT = 40
PIPE_WIDTH = 60
PIPE_GAP = 150  # gap between pipes
PIPE_SPACING = 300  # horizontal spacing between pipes

# Initialize Pygame
pygame.init()

class FlappyBirdEnv(gym.Env):
    def __init__(self, render_mode=False):
        super(FlappyBirdEnv, self).__init__()

        # Action space: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)
        
        # Observation space: (bird_y, bird_velocity, distances_to_next_pipes, gap_positions)
        low = np.array([0, -10] + [0]*3 + [0]*3)  # For 3 pipes
        high = np.array([SCREEN_HEIGHT, 10] + [SCREEN_WIDTH]*3 + [SCREEN_HEIGHT]*3)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Game state variables
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        self.pipes = []  # List to store pipe positions [(x, gap_y), ...]
        self.score = 0
        self.game_over = False
        self.render_mode = render_mode
        self.passed_pipes = set()  # Keep track of pipes we've passed

        if self.render_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

    def reset(self):
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        # Initialize 3 pipes with proper spacing
        self.pipes = [
            (SCREEN_WIDTH, random.randint(50, SCREEN_HEIGHT - PIPE_GAP - 50)),
            (SCREEN_WIDTH + PIPE_SPACING, random.randint(50, SCREEN_HEIGHT - PIPE_GAP - 50)),
            (SCREEN_WIDTH + PIPE_SPACING * 2, random.randint(50, SCREEN_HEIGHT - PIPE_GAP - 50))
        ]
        self.score = 0
        self.game_over = False
        self.passed_pipes = set()
        return self.get_observation()

    def get_observation(self):
        # Bird's Y position, velocity, distances to pipes, pipe gap positions
        pipe_distances = [p[0] for p in self.pipes]
        pipe_gaps = [p[1] for p in self.pipes]
        return np.array([self.bird_y, self.bird_velocity] + pipe_distances + pipe_gaps)

    def step(self, action):
        # Handle action
        if action == 1:  # Flap
            self.bird_velocity = -9
        else:  # accederate velocity down
            self.bird_velocity += 0.75  # random number gravity

        # Update bird's position
        self.bird_y += self.bird_velocity
        if self.bird_y > SCREEN_HEIGHT - BIRD_HEIGHT:
            self.bird_y = SCREEN_HEIGHT - BIRD_HEIGHT
        if self.bird_y < 0:
            self.bird_y = 0

        # Move pipes and check if new ones need to be added
        scored = False  # Track if the score is incremented in this step
        for i in range(len(self.pipes)):
            x, gap_y = self.pipes[i]
            x -= 3  # Move pipe left
            if x < -PIPE_WIDTH:
                # Move pipe to the right of the rightmost pipe
                rightmost_x = max(p[0] for p in self.pipes)
                x = rightmost_x + PIPE_SPACING
                gap_y = random.randint(50, SCREEN_HEIGHT - PIPE_GAP - 50)
                self.pipes[i] = (x, gap_y)
                # Reset this pipe's passed status
                if i in self.passed_pipes:
                    self.passed_pipes.remove(i)
            else:
                self.pipes[i] = (x, gap_y)

            # Check if the bird passes through the gap
            if 50 <= x <= 53:  # Only check in a small window when pipe center aligns with bird
                if i not in self.passed_pipes:  # Bird hasnt passed through this this pipe
                    self.score += 1
                    self.passed_pipes.add(i)
                    scored = True

        # Check for collisions with any pipe
        for pipe_x, pipe_gap_y in self.pipes:
            if (pipe_x < 50 + BIRD_WIDTH and pipe_x + PIPE_WIDTH > 50 and 
                (self.bird_y < pipe_gap_y or self.bird_y + BIRD_HEIGHT > pipe_gap_y + PIPE_GAP)):
                self.game_over = True

        if self.bird_y >= SCREEN_HEIGHT - BIRD_HEIGHT:
            self.game_over = True

        # Reward structure
        if self.game_over:
            reward = -10  # Negative reward for dying
        elif scored:
            print("Line 115: GAP")
            reward = 10  # Positive reward for passing through the gap
        else:
            reward = 0.1  # slight reward for just staying alive to prevent early flatline

        # Return observation, reward, done (game over?), info (empty here)
        done = self.game_over
        return self.get_observation(), reward, done, {}

    def render(self):
        if not self.render_mode:
            return
        self.screen.fill((255, 255, 255))  # White background
        # Draw the bird
        pygame.draw.rect(self.screen, (255, 0, 0), (50, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT))
        # Draw all pipes
        for pipe_x, pipe_gap_y in self.pipes:
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe_x, 0, PIPE_WIDTH, pipe_gap_y))
            pygame.draw.rect(self.screen, (0, 255, 0), 
                           (pipe_x, pipe_gap_y + PIPE_GAP, PIPE_WIDTH, 
                            SCREEN_HEIGHT - (pipe_gap_y + PIPE_GAP)))
        # Draw score
        font = pygame.font.SysFont('Arial', 30)
        score_text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (20, 20))
        # Refresh the screen
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS

    def close(self):
        if self.render_mode:
            pygame.quit()

def play_game():
    env = FlappyBirdEnv(render_mode=True)
    state = env.reset()
    done = False
    
    while not done:
        env.render()
        
        # Handle events
        action = 0  # Default action is do nothing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1  # Flap when spacebar is pressed
        
        state, reward, done, _ = env.step(action)
        
    print(f"Game Over! Final Score: {env.score}")
    env.close()

'''
if __name__ == "__main__":
    play_game()
'''
