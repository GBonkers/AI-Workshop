"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    X goes first; players alternate turns.
    """
    # Count number of X's and O's on the board
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    # If X and O have played same number of moves, it's X's turn; otherwise, it's O's turn
    return X if x_count == o_count else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    Actions are cells that are EMPTY.
    """
    possible_actions = set()
    for i in range(3):  # Iterate over rows
        for j in range(3):  # Iterate over columns
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    Does not modify the original board. Raises Exception if action is invalid.
    """
    (i, j) = action
    # Check action validity
    if board[i][j] is not EMPTY:
        raise Exception("Invalid action: Cell is not empty")
    # Create a deep copy of the board to avoid mutating original
    new_board = [row.copy() for row in board]
    # Apply move for current player
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    Checks rows, columns, and diagonals for three in a row.
    """
    # Check rows for a win
    for row in board:
        if row[0] == row[1] == row[2] and row[0] is not EMPTY:
            return row[0]
    # Check columns for a win
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] and board[0][j] is not EMPTY:
            return board[0][j]
    # Check diagonal (top-left to bottom-right)
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not EMPTY:
        return board[0][0]
    # Check diagonal (top-right to bottom-left)
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not EMPTY:
        return board[0][2]
    # No winner found
    return None


def terminal(board):
    """
    Returns True if game is over (win or tie), False otherwise.
    """
    # If there's a winner, game is over
    if winner(board) is not None:
        return True
    # If there is any empty cell, game is not over
    for row in board:
        if EMPTY in row:
            return False
    # No empty cells and no winner means tie
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    Assumes board is terminal.
    """
    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player using
    Minimax with alpha-beta pruning.
    """
    if terminal(board):
        return None

    current = player(board)
    alpha = -math.inf
    beta = math.inf
    best_action = None

    # If X's turn, maximize; else minimize
    if current == X:
        value = -math.inf
        for action in actions(board):
            # After X moves, it's O's turn: use min_value
            new_val = min_value(result(board, action), alpha, beta)
            # Update best if found higher value
            if new_val > value:
                value = new_val
                best_action = action
            alpha = max(alpha, value)
        return best_action
    else:
        value = math.inf
        for action in actions(board):
            # After O moves, it's X's turn: use max_value
            new_val = max_value(result(board, action), alpha, beta)
            if new_val < value:
                value = new_val
                best_action = action
            beta = min(beta, value)
        return best_action

def max_value(board, alpha, beta):
    """
    Returns the maximum utility value achievable from this board.
    Uses alpha-beta pruning to cut off branches.
    """
    if terminal(board):
        return utility(board)
    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action), alpha, beta))
        # Prune: if v is already >= beta, no need to explore further
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v


def min_value(board, alpha, beta):
    """
    Returns the minimum utility value achievable from this board.
    Uses alpha-beta pruning to cut off branches.
    """
    if terminal(board):
        return utility(board)
    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action), alpha, beta))
        # Prune: if v is already <= alpha, no need to explore further
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v
