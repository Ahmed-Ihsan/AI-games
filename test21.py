import pygame
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class TrainingMetrics:
    episode_rewards: List[float]
    episode_steps: List[int]
    success_rate: List[float]

class QLearningAgent:
    def __init__(self, 
                 map_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        self.map_size = map_size
        self.state_size = map_size * map_size
        self.action_size = 4
        self.q_table = np.zeros((self.state_size, self.action_size))
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_state_index(self, position: Tuple[int, int]) -> int:
        return position[0] * self.map_size + position[1]
    
    def get_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])
    
    def take_action(self, position: Tuple[int, int], action: int) -> Tuple[int, int]:
        new_pos = (position[0] + self.moves[action][0], 
                  position[1] + self.moves[action][1])
        
        if 0 <= new_pos[0] < self.map_size and 0 <= new_pos[1] < self.map_size:
            return new_pos
        return position
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        best_next_value = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        
        # Q-learning update formula
        new_q = current_q + self.lr * (reward + self.gamma * best_next_value - current_q)
        self.q_table[state][action] = new_q

class QLearnGame:
    def __init__(self, map_size: int = 5):
        pygame.init()
        self.map_size = map_size
        self.cell_size = 80
        self.width = map_size * self.cell_size
        self.height = map_size * self.cell_size + 100  # Extra space for metrics
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Q-Learning Pathfinding")
        
        self.agent = QLearningAgent(map_size)
        self.target_pos = (map_size-1, map_size-1)
        self.current_pos = (0, 0)
        
        # Initialize fonts
        pygame.font.init()
        self.font = pygame.font.Font(None, 36)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        # Metrics
        self.metrics = TrainingMetrics([], [], [])
    
    def draw_grid(self, episode: int, total_reward: float):
        self.screen.fill(self.WHITE)
        
        # Draw grid
        for x in range(self.map_size):
            for y in range(self.map_size):
                rect = pygame.Rect(y*self.cell_size, x*self.cell_size, 
                                 self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)
        
        # Draw target
        target_rect = pygame.Rect(self.target_pos[1]*self.cell_size, 
                                self.target_pos[0]*self.cell_size,
                                self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.GREEN, target_rect)
        
        # Draw agent
        agent_rect = pygame.Rect(self.current_pos[1]*self.cell_size,
                               self.current_pos[0]*self.cell_size,
                               self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.RED, agent_rect)
        
        # Draw metrics
        metrics_text = f"Episode: {episode} | Reward: {total_reward:.1f} | ε: {self.agent.epsilon:.3f}"
        text_surface = self.font.render(metrics_text, True, self.BLACK)
        self.screen.blit(text_surface, (10, self.height - 50))
        
        pygame.display.flip()
    
    def plot_metrics(self):
        plt.figure(figsize=(15, 5))
        
        # Plot rewards
        plt.subplot(131)
        plt.plot(self.metrics.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        # Plot steps
        plt.subplot(132)
        plt.plot(self.metrics.episode_steps)
        plt.title('Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        # Plot success rate
        plt.subplot(133)
        plt.plot(self.metrics.success_rate)
        plt.title('Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        
        plt.tight_layout()
        plt.show()
    
    def train(self, episodes: int = 1000, max_steps: int = 200):
        success_window = []
        
        for episode in range(episodes):
            self.current_pos = (0, 0)  # Start from fixed position
            total_reward = 0
            steps = 0
            
            while steps < max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                current_state = self.agent.get_state_index(self.current_pos)
                action = self.agent.get_action(current_state)
                new_pos = self.agent.take_action(self.current_pos, action)
                new_state = self.agent.get_state_index(new_pos)
                
                # Reward shaping
                reward = -1  # Step penalty
                if new_pos == self.target_pos:
                    reward = 100  # Goal reward
                elif new_pos == self.current_pos:
                    reward = -5  # Wall collision penalty
                
                total_reward += reward
                self.agent.update(current_state, action, reward, new_state)
                self.current_pos = new_pos
                steps += 1
                
                self.draw_grid(episode, total_reward)
                sleep(0.01)  # Visualization delay
                
                if self.current_pos == self.target_pos:
                    success_window.append(1)
                    break
            
            if steps == max_steps:
                success_window.append(0)
            
            # Update metrics
            self.metrics.episode_rewards.append(total_reward)
            self.metrics.episode_steps.append(steps)
            
            # Calculate success rate over last 100 episodes
            if len(success_window) > 100:
                success_window.pop(0)
            self.metrics.success_rate.append(sum(success_window) / len(success_window))
            
            self.agent.decay_epsilon()
    
    def run(self, episodes: int = 1000):
        self.train(episodes=episodes)
        self.plot_metrics()
        pygame.quit()

if __name__ == "__main__":
    game = QLearnGame(map_size=10)
    game.run(episodes=10000)