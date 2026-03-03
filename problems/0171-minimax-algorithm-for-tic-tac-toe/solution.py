import numpy as np

def is_winner(board, player):
    for i in range(3):
        if all(board[i, j] == player for j in range(3)): return True
        if all(board[j, i] == player for j in range(3)): return True
    if all(board[i, i] == player for i in range(3)): return True
    if all(board[i, 2-i] == player for i in range(3)): return True
    return False

def is_full(board):
    return not any(board[i, j] == '' for i in range(3) for j in range(3))

def get_available_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == '']

def minimax_tictactoe(board: np.ndarray, player: str) -> tuple:
    opponent = 'O' if player == 'X' else 'X'

    def minimax(cur_player):
        # terminal checks must consider BOTH players
        if is_winner(board, player):      # AI player wins
            return 1
        if is_winner(board, opponent):    # opponent wins
            return -1
        if is_full(board):
            return 0

        moves = get_available_mov