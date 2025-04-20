import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
import time
import os
from heapq import heappush, heappop

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 600
GRID_SIZE = 10
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
FPS = 10

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        features = self.features(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

class PPO:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.epsilon = 0.2
        self.gae_lambda = 0.95
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.0003)
    
    def get_action(self, state):
        state = torch.FloatTensor(state)
        action_probs, _ = self.actor_critic(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def compute_gae(self, rewards, values, done):
        gae = 0
        returns = []
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = 0 if done else values[step]
            else:
                next_value = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae
            returns.insert(0, gae + values[step])
        
        return returns
    
    def update(self, states, actions, old_log_probs, rewards, values, done):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.stack(old_log_probs)
        values = torch.FloatTensor(values).reshape(-1, 1)
        
        returns = torch.FloatTensor(self.compute_gae(rewards, values.detach().numpy().flatten(), done))
        returns = returns.reshape(-1, 1)
        advantages = returns - values
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(10):
            action_probs, current_values = self.actor_critic(states)
            dist = Categorical(action_probs)
            current_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(current_log_probs - old_log_probs.detach())
            ratio = ratio.reshape(-1, 1)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(current_values, returns)
            
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
            self.optimizer.step()

class AStar:
    def __init__(self, grid_size):
        self.grid_size = grid_size
    
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos, obstacles):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < self.grid_size and 
                0 <= new_pos[1] < self.grid_size and 
                new_pos not in obstacles):
                neighbors.append(new_pos)
        return neighbors
    
    def find_path(self, start, goal, obstacles):
        start = tuple(start)
        goal = tuple(goal)
        
        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heappop(frontier)[1]
            
            if current == goal:
                break
            
            for next_pos in self.get_neighbors(current, obstacles):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        
        return path if path[0] == start else []

class GridWorld:
    def __init__(self):
        self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("PPO Grid World with A* Teaching")
        self.clock = pygame.time.Clock()
        
        self.action_space = 4
        self.state_dim = 4
        
        self.max_steps = GRID_SIZE * 3
        self.current_steps = 0
        
        self.astar = AStar(GRID_SIZE)
        self.optimal_path = None
        self.current_path_index = 0
        
        self.generate_random_map()
    
    def generate_random_map(self):
        self.obstacles = set()
        num_obstacles = random.randint(GRID_SIZE * GRID_SIZE // 6, GRID_SIZE * GRID_SIZE // 4)
        
        while len(self.obstacles) < num_obstacles:
            pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if pos != (0, 0) and pos != (0, 1) and pos != (1, 0):
                self.obstacles.add(pos)
        
        while True:
            self.goal = (random.randint(GRID_SIZE//2, GRID_SIZE-1), 
                        random.randint(GRID_SIZE//2, GRID_SIZE-1))
            if self.goal not in self.obstacles:
                break
        
        self.agent_pos = [0, 0]
        
        if not self.check_path_exists():
            self.generate_random_map()
    
    def check_path_exists(self):
        return bool(self.astar.find_path(self.agent_pos, self.goal, self.obstacles))
    
    def get_optimal_action(self):
        if not self.optimal_path or self.current_path_index >= len(self.optimal_path) - 1:
            return None
        
        current = self.optimal_path[self.current_path_index]
        next_pos = self.optimal_path[self.current_path_index + 1]
        
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]
        
        if dx == 1: return 2  # down
        if dx == -1: return 0  # up
        if dy == 1: return 1  # right
        if dy == -1: return 3  # left
        
        return None
    
    def get_state(self):
        return np.array([
            self.agent_pos[0] / GRID_SIZE,
            self.agent_pos[1] / GRID_SIZE,
            self.goal[0] / GRID_SIZE,
            self.goal[1] / GRID_SIZE
        ])
    
    def reset(self):
        self.current_steps = 0
        self.generate_random_map()
        self.agent_pos = [0, 0]
        self.optimal_path = self.astar.find_path(self.agent_pos, self.goal, self.obstacles)
        self.current_path_index = 0
        return self.get_state()
    
    def step(self, action):
        self.current_steps += 1
        
        if self.current_steps >= self.max_steps:
            return self.get_state(), -10, True
        
        reward = -0.1
        old_pos = self.agent_pos.copy()
        
        if action == 0:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # right
            self.agent_pos[1] = min(GRID_SIZE - 1, self.agent_pos[1] + 1)
        elif action == 2:  # down
            self.agent_pos[0] = min(GRID_SIZE - 1, self.agent_pos[0] + 1)
        elif action == 3:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        
        if tuple(self.agent_pos) in self.obstacles:
            self.agent_pos = old_pos
            reward = -1
        elif tuple(self.agent_pos) == self.goal:
            reward = 100
            done = True
            return self.get_state(), reward, done
        
        current_dist = abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])
        old_dist = abs(old_pos[0] - self.goal[0]) + abs(old_pos[1] - self.goal[1])
        reward += 0.1 * (old_dist - current_dist)
        
        if self.optimal_path and tuple(self.agent_pos) in self.optimal_path:
            self.current_path_index = self.optimal_path.index(tuple(self.agent_pos))
        
        done = False
        return self.get_state(), reward, done
    
    def render(self):
        self.window.fill(WHITE)
        
        for i in range(GRID_SIZE):
            pygame.draw.line(self.window, GRAY, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE))
            pygame.draw.line(self.window, GRAY, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE))
        
        if self.optimal_path:
            for i in range(len(self.optimal_path) - 1):
                start_pos = self.optimal_path[i]
                end_pos = self.optimal_path[i + 1]
                pygame.draw.line(self.window, RED,
                               (start_pos[1] * CELL_SIZE + CELL_SIZE//2,
                                start_pos[0] * CELL_SIZE + CELL_SIZE//2),
                               (end_pos[1] * CELL_SIZE + CELL_SIZE//2,
                                end_pos[0] * CELL_SIZE + CELL_SIZE//2), 2)
        
        for obs in self.obstacles:
            pygame.draw.rect(self.window, BLACK,
                           (obs[1] * CELL_SIZE, obs[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        pygame.draw.rect(self.window, GREEN,
                        (self.goal[1] * CELL_SIZE, self.goal[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        pygame.draw.circle(self.window, BLUE,
                         (self.agent_pos[1] * CELL_SIZE + CELL_SIZE//2,
                          self.agent_pos[0] * CELL_SIZE + CELL_SIZE//2),
                         CELL_SIZE//3)
        
        pygame.display.flip()

def train_ppo_with_astar(env, max_episodes=1000, teaching_episodes=200):
    ppo = PPO(state_dim=env.state_dim, action_dim=env.action_space)
    
    best_reward = float('-inf')
    episode_rewards = []
    
    try:
        for episode in range(max_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            states, actions, log_probs, rewards, values = [], [], [], [], []
            
            while not done:
                state_tensor = torch.FloatTensor(state)
                action_probs, value = ppo.actor_critic(state_tensor)
                
                if episode < teaching_episodes and random.random() < 0.7:
                    optimal_action = env.get_optimal_action()
                    if optimal_action is not None:
                        action = optimal_action
                        log_prob = torch.log(action_probs[action])
                    else:
                        action, log_prob = ppo.get_action(state)
                else:
                    action, log_prob = ppo.get_action(state)
                
                next_state, reward, done = env.step(action)
                
                if episode < teaching_episodes and env.optimal_path:
                    if tuple(env.agent_pos) in env.optimal_path:
                        reward += 0.5
                
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value.item())
                
                state = next_state
                episode_reward += reward
                
                env.render()
                env.clock.tick(FPS)
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
            
            episode_rewards.append(episode_reward)
            ppo.update(states, actions, log_probs, rewards, values, done)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(ppo.actor_critic.state_dict(), 'best_model.pth')
            
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode}, Average Reward (last 10): {avg_reward:.2f}")
                if episode < teaching_episodes:
                    print(f"Currently in teaching phase: {episode}/{teaching_episodes}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    
    return ppo

def test_ppo(env, ppo, num_episodes=5):
    print("\nTesting trained agent...")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 100:
            state_tensor = torch.FloatTensor(env.get_state())
            action_probs, _ = ppo.actor_critic(state_tensor)
            action = torch.argmax(action_probs).item()
            
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            env.render()
            time.sleep(0.1)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
        
        print(f"Episode {episode + 1}: Steps: {steps}, Reward: {total_reward}")
        time.sleep(1)

if __name__ == "__main__":
    env = GridWorld()
    ppo_agent = train_ppo_with_astar(env, max_episodes=500, teaching_episodes=200)
    test_ppo(env, ppo_agent)
    pygame.quit()