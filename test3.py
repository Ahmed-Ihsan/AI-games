import pygame
import random
import sys
import math
from collections import deque
import copy

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 40
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARK_GREEN = (0, 200, 0)
DARK_BLUE = (0, 0, 200)

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Snake vs Snake with Minimax AI')

# Directional constants
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

class Snake:
    def __init__(self, start_pos, color, dark_color):
        self.positions = [start_pos]
        self.direction = random.choice([LEFT, RIGHT])
        self.color = color
        self.dark_color = dark_color
        self.alive = True
        self.score = 0
        self.last_direction = self.direction

    def clone(self):
        new_snake = Snake(self.positions[0], self.color, self.dark_color)
        new_snake.positions = self.positions.copy()
        new_snake.direction = self.direction
        new_snake.last_direction = self.last_direction
        new_snake.alive = self.alive
        new_snake.score = self.score
        return new_snake

    def get_head_position(self):
        return self.positions[0]

    def set_direction(self, new_direction):
        if (new_direction[0] != -self.last_direction[0] or 
            new_direction[1] != -self.last_direction[1]):
            self.direction = new_direction

    def update(self, opponent_positions):
        if not self.alive:
            return False

        cur = self.get_head_position()
        x, y = self.direction
        new = (cur[0] + x, cur[1] + y)
        
        if (new[0] < 0 or new[0] >= GRID_WIDTH or 
            new[1] < 0 or new[1] >= GRID_HEIGHT):
            self.alive = False
            return False

        if (new in self.positions[1:] or 
            new in opponent_positions):
            self.alive = False
            return False
        
        self.positions.insert(0, new)
        self.positions.pop()
        self.last_direction = self.direction
        return True

    def grow(self):
        self.positions.append(self.positions[-1])
        self.score += 1

    def render(self, surface):
        for i, p in enumerate(self.positions):
            color = self.dark_color if i == 0 else self.color
            pygame.draw.rect(surface, color, 
                           (p[0] * GRID_SIZE, p[1] * GRID_SIZE, 
                            GRID_SIZE - 2, GRID_SIZE - 2))

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED

    def randomize_position(self, snake1_pos, snake2_pos):
        available_positions = [
            (x, y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT)
            if (x, y) not in snake1_pos and (x, y) not in snake2_pos
        ]
        if available_positions:
            self.position = random.choice(available_positions)
            return True
        return False

    def render(self, surface):
        pygame.draw.rect(surface, self.color,
                        (self.position[0] * GRID_SIZE,
                         self.position[1] * GRID_SIZE,
                         GRID_SIZE - 2, GRID_SIZE - 2))

def evaluate_position(snake_ai, snake_player, food):
    if not snake_ai.alive:
        return -1000
    if not snake_player.alive:
        return 1000
    
    ai_head = snake_ai.get_head_position()
    player_head = snake_player.get_head_position()
    ai_food_dist = abs(ai_head[0] - food.position[0]) + abs(ai_head[1] - food.position[1])
    player_food_dist = abs(player_head[0] - food.position[0]) + abs(player_head[1] - food.position[1])
    
    score = player_food_dist - ai_food_dist
    
    ai_spaces = count_reachable_spaces(ai_head, snake_ai.positions + snake_player.positions)
    score += ai_spaces * 2
    
    return score

def count_reachable_spaces(start, blocked_positions):
    visited = set()
    queue = deque([start])
    
    while queue:
        pos = queue.popleft()
        if pos in visited:
            continue
        visited.add(pos)
        
        for direction in DIRECTIONS:
            new_pos = (pos[0] + direction[0], pos[1] + direction[1])
            if (0 <= new_pos[0] < GRID_WIDTH and 
                0 <= new_pos[1] < GRID_HEIGHT and 
                new_pos not in blocked_positions and 
                new_pos not in visited):
                queue.append(new_pos)
    
    return len(visited)

def get_valid_moves(snake):
    return [d for d in DIRECTIONS if d[0] != -snake.last_direction[0] or d[1] != -snake.last_direction[1]]

def simulate_move(snake, direction, opponent_positions):
    new_snake = snake.clone()
    new_snake.direction = direction
    new_snake.update(opponent_positions)
    return new_snake

def minimax(snake_ai, snake_player, food, depth, alpha, beta, maximizing_player):
    if depth == 0 or not snake_ai.alive or not snake_player.alive:
        return evaluate_position(snake_ai, snake_player, food)
    
    if maximizing_player:
        max_eval = -math.inf
        for direction in get_valid_moves(snake_ai):
            new_snake_ai = simulate_move(snake_ai, direction, snake_player.positions)
            if new_snake_ai.alive:
                eval = minimax(new_snake_ai, snake_player, food, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return max_eval
    else:
        min_eval = math.inf
        for direction in get_valid_moves(snake_player):
            new_snake_player = simulate_move(snake_player, direction, snake_ai.positions)
            if new_snake_player.alive:
                eval = minimax(snake_ai, new_snake_player, food, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        return min_eval

def get_best_move(snake_ai, snake_player, food):
    best_score = -math.inf
    best_move = snake_ai.direction
    
    for direction in get_valid_moves(snake_ai):
        new_snake_ai = simulate_move(snake_ai, direction, snake_player.positions)
        if new_snake_ai.alive:
            score = minimax(new_snake_ai, snake_player, food, 2, -math.inf, math.inf, False)
            if score > best_score:
                best_score = score
                best_move = direction
    
    return best_move

def main():
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    snake_player = Snake((GRID_WIDTH // 4, GRID_HEIGHT // 2), GREEN, DARK_GREEN)
    snake_ai = Snake((3 * GRID_WIDTH // 4, GRID_HEIGHT // 2), BLUE, DARK_BLUE)
    food = Food()
    food.randomize_position(snake_player.positions, snake_ai.positions)
    
    running = True
    game_over = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif not game_over:
                    if event.key == pygame.K_UP:
                        snake_player.set_direction(UP)
                    elif event.key == pygame.K_DOWN:
                        snake_player.set_direction(DOWN)
                    elif event.key == pygame.K_LEFT:
                        snake_player.set_direction(LEFT)
                    elif event.key == pygame.K_RIGHT:
                        snake_player.set_direction(RIGHT)
                elif event.key == pygame.K_SPACE:
                    snake_player = Snake((GRID_WIDTH // 4, GRID_HEIGHT // 2), GREEN, DARK_GREEN)
                    snake_ai = Snake((3 * GRID_WIDTH // 4, GRID_HEIGHT // 2), BLUE, DARK_BLUE)
                    food.randomize_position(snake_player.positions, snake_ai.positions)
                    game_over = False
        
        if not game_over:
            if snake_ai.alive:
                snake_ai.direction = get_best_move(snake_ai, snake_player, food)
            
            snake_player.update(snake_ai.positions)
            snake_ai.update(snake_player.positions)
            
            if snake_player.get_head_position() == food.position:
                snake_player.grow()
                if not food.randomize_position(snake_player.positions, snake_ai.positions):
                    game_over = True
            if snake_ai.get_head_position() == food.position:
                snake_ai.grow()
                if not food.randomize_position(snake_player.positions, snake_ai.positions):
                    game_over = True
            
            if not snake_player.alive or not snake_ai.alive:
                game_over = True
        
        screen.fill(BLACK)
        snake_player.render(screen)
        snake_ai.render(screen)
        food.render(screen)
        
        player_score = font.render(f'Player: {snake_player.score}', True, GREEN)
        ai_score = font.render(f'AI: {snake_ai.score}', True, BLUE)
        screen.blit(player_score, (10, 10))
        screen.blit(ai_score, (10, 50))
        
        if game_over:
            game_over_text = font.render('Game Over!', True, WHITE)
            winner_text = font.render(
                'Player Wins!' if not snake_ai.alive else 'AI Wins!' if not snake_player.alive else 'Draw!',
                True, WHITE
            )
            restart_text = font.render('Press SPACE to restart or ESC to quit', True, WHITE)
            screen.blit(game_over_text, (WINDOW_WIDTH//2 - 100, WINDOW_HEIGHT//2 - 50))
            screen.blit(winner_text, (WINDOW_WIDTH//2 - 100, WINDOW_HEIGHT//2))
            screen.blit(restart_text, (WINDOW_WIDTH//2 - 200, WINDOW_HEIGHT//2 + 50))
        
        pygame.display.update()
        clock.tick(8)
    
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()