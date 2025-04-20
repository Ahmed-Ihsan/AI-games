import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH = 600
HEIGHT = 600
LINE_WIDTH = 15
BOARD_ROWS = 3
BOARD_COLS = 3
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4

# Colors
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe')
screen.fill(BG_COLOR)

# Board
board = [['' for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]

def draw_lines():
    # Horizontal lines
    pygame.draw.line(screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (0, 2 * SQUARE_SIZE), (WIDTH, 2 * SQUARE_SIZE), LINE_WIDTH)
    # Vertical lines
    pygame.draw.line(screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, HEIGHT), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (2 * SQUARE_SIZE, 0), (2 * SQUARE_SIZE, HEIGHT), LINE_WIDTH)

def draw_figures():
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 'O':
                pygame.draw.circle(screen, CIRCLE_COLOR,
                                 (int(col * SQUARE_SIZE + SQUARE_SIZE // 2),
                                  int(row * SQUARE_SIZE + SQUARE_SIZE // 2)),
                                 CIRCLE_RADIUS, CIRCLE_WIDTH)
            elif board[row][col] == 'X':
                pygame.draw.line(screen, CROSS_COLOR,
                               (col * SQUARE_SIZE + SPACE,
                                row * SQUARE_SIZE + SQUARE_SIZE - SPACE),
                               (col * SQUARE_SIZE + SQUARE_SIZE - SPACE,
                                row * SQUARE_SIZE + SPACE), CROSS_WIDTH)
                pygame.draw.line(screen, CROSS_COLOR,
                               (col * SQUARE_SIZE + SPACE,
                                row * SQUARE_SIZE + SPACE),
                               (col * SQUARE_SIZE + SQUARE_SIZE - SPACE,
                                row * SQUARE_SIZE + SQUARE_SIZE - SPACE), CROSS_WIDTH)

def mark_square(row, col, player):
    board[row][col] = player

def available_square(row, col):
    return board[row][col] == ''

def is_board_full():
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == '':
                return False
    return True

def check_win(player):
    # Vertical win check
    for col in range(BOARD_COLS):
        if board[0][col] == player and board[1][col] == player and board[2][col] == player:
            return True

    # Horizontal win check
    for row in range(BOARD_ROWS):
        if board[row][0] == player and board[row][1] == player and board[row][2] == player:
            return True

    # Ascending diagonal win check
    if board[2][0] == player and board[1][1] == player and board[0][2] == player:
        return True

    # Descending diagonal win check
    if board[0][0] == player and board[1][1] == player and board[2][2] == player:
        return True

    return False

def minimax(depth, is_maximizing):
    if check_win('O'):
        return 1
    if check_win('X'):
        return -1
    if is_board_full():
        return 0

    if is_maximizing:
        best_score = -math.inf
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if board[row][col] == '':
                    board[row][col] = 'O'
                    score = minimax(depth + 1, False)
                    board[row][col] = ''
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if board[row][col] == '':
                    board[row][col] = 'X'
                    score = minimax(depth + 1, True)
                    board[row][col] = ''
                    best_score = min(score, best_score)
        return best_score

def get_best_move():
    best_score = -math.inf
    best_move = None
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == '':
                board[row][col] = 'O'
                score = minimax(0, False)
                board[row][col] = ''
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
    return best_move

# Main game loop
game_over = False
player_turn = True  # True for X, False for O (AI)

draw_lines()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
            mouseX = event.pos[0]
            mouseY = event.pos[1]

            clicked_row = int(mouseY // SQUARE_SIZE)
            clicked_col = int(mouseX // SQUARE_SIZE)

            if player_turn:
                if available_square(clicked_row, clicked_col):
                    mark_square(clicked_row, clicked_col, 'X')
                    if check_win('X'):
                        game_over = True
                    player_turn = False
                    draw_figures()

        # AI turn
        if not player_turn and not game_over:
            ai_row, ai_col = get_best_move()
            mark_square(ai_row, ai_col, 'O')
            if check_win('O'):
                game_over = True
            player_turn = True
            draw_figures()

        if is_board_full():
            game_over = True

    pygame.display.update()