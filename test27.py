import pygame
import numpy as np
import random
from collections import defaultdict
import json
import os

class SnakeGame:
    def __init__(self, width=400, height=400):
        pygame.init()
        self.width = width
        self.height = height
        self.grid_size = 20
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.reset()
    
    def reset(self):
        self.snake = [(self.width//2, self.height//2)]
        self.direction = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        self.food = self.spawn_food()
        self.score = 0
        self.steps = 0
        return self.get_state()
    
    def spawn_food(self):
        while True:
            pos = (random.randint(0, self.width//self.grid_size-1) * self.grid_size,
                  random.randint(0, self.height//self.grid_size-1) * self.grid_size)
            if pos not in self.snake:
                return pos
    
    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        food_dir_x = (food_x > head_x) - (food_x < head_x)
        food_dir_y = (food_y > head_y) - (food_y < head_y)
        
        danger_straight = self.is_collision((head_x + self.direction[0]*self.grid_size, 
                                          head_y + self.direction[1]*self.grid_size))
        danger_right = self.is_collision((head_x + self.rotate_right(self.direction)[0]*self.grid_size,
                                        head_y + self.rotate_right(self.direction)[1]*self.grid_size))
        danger_left = self.is_collision((head_x + self.rotate_left(self.direction)[0]*self.grid_size,
                                       head_y + self.rotate_left(self.direction)[1]*self.grid_size))
        
        return (food_dir_x, food_dir_y, danger_straight, danger_right, danger_left,
                self.direction[0], self.direction[1])
    
    def rotate_right(self, dir):
        return (-dir[1], dir[0])
    
    def rotate_left(self, dir):
        return (dir[1], -dir[0])
    
    def is_collision(self, pos):
        x, y = pos
        return (x < 0 or x >= self.width or 
                y < 0 or y >= self.height or 
                pos in self.snake[:-1])
    
    def step(self, action):
        if action == 1:
            self.direction = self.rotate_right(self.direction)
        elif action == 2:
            self.direction = self.rotate_left(self.direction)
            
        new_head = (self.snake[0][0] + self.direction[0]*self.grid_size,
                   self.snake[0][1] + self.direction[1]*self.grid_size)
        
        self.steps += 1
        reward = 0
        done = False
        
        if self.is_collision(new_head):
            reward = -10
            done = True
        else:
            self.snake.insert(0, new_head)
            
            if new_head == self.food:
                self.food = self.spawn_food()
                reward = 10
                self.score += 1
            else:
                self.snake.pop()
                reward = -0.1
                
            if self.steps > 100*len(self.snake):
                done = True
        
        return self.get_state(), reward, done
    
    def render(self):
        self.screen.fill((0, 0, 0))
        
        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0),
                           (segment[0], segment[1], self.grid_size-2, self.grid_size-2))
        
        pygame.draw.rect(self.screen, (255, 0, 0),
                        (self.food[0], self.food[1], self.grid_size-2, self.grid_size-2))
        
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(60)

class SarsaAgent:
    def __init__(self, actions=3):
        self.q_table = defaultdict(lambda: np.zeros(actions))
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.001
        self.gamma = 0.9
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename='snake_agent.json'):
        # Convert Q-table to regular dictionary with string keys
        q_dict = {str(state): values.tolist() for state, values in self.q_table.items()}
        
        # Save agent parameters and Q-table
        save_data = {
            'q_table': q_dict,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f)
        
        print(f"Agent saved to {filename}")
    
    def load(self, filename='snake_agent.json'):
        if not os.path.exists(filename):
            print(f"No saved agent found at {filename}")
            return False
        
        with open(filename, 'r') as f:
            save_data = json.load(f)
        
        # Load parameters
        self.epsilon = save_data['epsilon']
        self.epsilon_min = save_data['epsilon_min']
        self.epsilon_decay = save_data['epsilon_decay']
        self.alpha = save_data['alpha']
        self.gamma = save_data['gamma']
        
        # Convert loaded Q-table back to defaultdict with tuple keys
        q_dict = save_data['q_table']
        self.q_table = defaultdict(lambda: np.zeros(3))
        for state_str, values in q_dict.items():
            state_tuple = eval(state_str)  # Convert string representation back to tuple
            self.q_table[state_tuple] = np.array(values)
        
        print(f"Agent loaded from {filename}")
        return True

def train(load_existing=True, save_interval=1000):
    env = SnakeGame()
    agent = SarsaAgent()
    
    # Try to load existing agent
    if load_existing:
        agent.load()
    
    episodes = 1000000
    best_score = 0
    
    try:
        for episode in range(episodes):
            state = env.reset()
            action = agent.get_action(state)
            total_reward = 0
            done = False
            
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        agent.save()  # Save before quitting
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            agent.save()  # Save before quitting
                            pygame.quit()
                            return
                
                next_state, reward, done = env.step(action)
                next_action = agent.get_action(next_state) if not done else 0
                
                agent.learn(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
                total_reward += reward
                
                env.render()
                
            if env.score > best_score:
                best_score = env.score
                print(f"Episode {episode}, New Best Score: {best_score}, Epsilon: {agent.epsilon:.3f}")
            
            # Save periodically
            if episode > 0 and episode % save_interval == 0:
                agent.save()
                agent.save()  # Also save to default filename
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save()  # Save on interrupt

if __name__ == "__main__":
    train()