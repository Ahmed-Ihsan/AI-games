import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import math
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class FPSEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Initialize pygame for visualization
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        
        # Define action space (continuous: delta_x, delta_y for aim movement)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )
        
        # Define observation space
        # [agent_x, agent_y, target_x, target_y, relative_angle]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -math.pi]),
            high=np.array([self.width, self.height, self.width, self.height, math.pi]),
            dtype=np.float32
        )
        
        # Game state
        self.agent_pos = None
        self.target_pos = None
        self.steps = 0
        self.max_steps = 200
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Reset agent and target positions
        self.agent_pos = np.array([self.width/2, self.height/2])
        self.target_pos = self.generate_target_position()
        self.steps = 0
        
        return self._get_observation(), {}
    
    def generate_target_position(self):
        return np.array([
            np.random.uniform(50, self.width-50),
            np.random.uniform(50, self.height-50)
        ])
    
    def _get_observation(self):
        # Calculate relative angle between agent and target
        dx = self.target_pos[0] - self.agent_pos[0]
        dy = self.target_pos[1] - self.agent_pos[1]
        angle = math.atan2(dy, dx)
        
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.target_pos[0],
            self.target_pos[1],
            angle
        ])
    
    def step(self, action):
        self.steps += 1
        
        # Update agent position based on action
        movement_scale = 10.0
        delta_pos = action * movement_scale
        self.agent_pos += delta_pos
        
        # Clip position to screen bounds
        self.agent_pos = np.clip(
            self.agent_pos,
            [0, 0],
            [self.width, self.height]
        )
        
        # Calculate distance to target
        distance = np.linalg.norm(self.target_pos - self.agent_pos)
        
        # Define rewards
        reward = 0
        terminated = False
        
        # Hit detection (within 20 pixels)
        if distance < 20:
            reward = 100
            terminated = True
        else:
            # Shaped reward based on distance
            reward = -distance / 100
        
        # Check if episode should end
        if self.steps >= self.max_steps:
            terminated = True
        
        # Render if display is active
        self._render_frame()
        
        return self._get_observation(), reward, terminated, False, {}
    
    def _render_frame(self):
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw target
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            self.target_pos.astype(int),
            20
        )
        
        # Draw agent
        pygame.draw.circle(
            self.screen,
            (0, 255, 0),
            self.agent_pos.astype(int),
            10
        )
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit()

def train_agent():
    # Create environment
    env = DummyVecEnv([lambda: FPSEnvironment()])
    
    # Initialize agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99
    )
    
    # Train the agent
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
    model.save("fps_agent")
    
    return model

def evaluate_agent(model, num_episodes=10):
    env = FPSEnvironment()
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            # Handle pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
        
        print(f"Episode {episode + 1} reward: {episode_reward}")
    
    env.close()

if __name__ == "__main__":
    # Train the agent
    trained_model = train_agent()
    
    # Evaluate the trained agent
    evaluate_agent(trained_model)