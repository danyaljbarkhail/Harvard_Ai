"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    #Returns starting state of the board. Tested.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    #Returns player who has the next turn on a board. Tested.
    """
    X_counter = 0
    O_counter = 0

    for row in board:
        for col in row:
            if col == X:
                X_counter += 1
            elif col == O:
                O_counter += 1
    
    if X_counter > O_counter:
        return O
    else:
        return X


def actions(board):
    """
    #Returns set of all possible actions (i, j) available on the board. Tested.
    """
    possible_actions = set()

    for i in range(0, len(board)):
        for j in range(0, len(board[0])):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    
    return possible_actions


def result(board, action):
    """
    #Returns the board that results from making move (i, j) on the board.
    """
    
    # Create a copy of the board
    resultant_board = copy.deepcopy(board)
    
    if resultant_board[action[0]][action[1]] == EMPTY:
        resultant_board[action[0]][action[1]] = player(resultant_board)
    else:
        raise Exception("Illegal move.")

    return resultant_board



def winner(board):
    """
    #Returns the winner of the game, if there is one. Tested.
    """
    # Check horizontal
    for row in board:
        if len(set(iter(row))) == 1:
            return row[0]
    
    # Check vertical
    for col in range(len(board)):
        if board[0][col] == board[1][col] == board[2][col]:
            return board[0][col]
    
    # Check diagonal
    if board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    elif board[2][0] == board[1][1] == board[0][2]:
        return board[2][0]
    
    return None


def terminal(board):
    """
    #Returns True if game is over, False otherwise. Tested.
    """
    # Check if won
    if winner(board) is not None:
        return True
    
    # Check if tied
    contains_EMPTY = False

    for row in board:
        for col in row:
             if col == EMPTY:
                contains_EMPTY = True
    
    if not contains_EMPTY:
        return True
    
    # Not terminal board if not won and not tied
    return False


def utility(board):
    """
    #Returns 1 if X has won the game, -1 if O has won, 0 otherwise. Tested.
    """
    if terminal(board):
        if winner(board) == X:
            return 1
        elif winner(board) == O:
            return -1
        else:
            return 0


def minimax(board):
    """
    #Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    # For max player, return action which produces
    # the highest utility after min player plays
    elif player(board) == X:
        value, move = max_value(board)

    elif player(board) == O:
        value, move = min_value(board)

    return move

def max_value(board):
    """
    #Calculate the maximum value of a state.
    """
    # Check if terminal state
    if terminal(board):
        return utility(board), None
    
    v = float('-inf')
    optimal_action = None

    for action in actions(board):
        final_value, opponent_move = min_value(result(board, action))

        if final_value > v:
            v = final_value
            optimal_action = action
            if v == 1:
                return v, optimal_action

    return v, optimal_action


def min_value(board):
    """
    #Calculate the minimum value of a state.
    """
    # Check if terminal state
    if terminal(board):
        return utility(board), None
    
    v = float('inf')
    optimal_action = None

    for action in actions(board):
        final_value, opponent_move = max_value(result(board, action))

        if final_value < v:
            v = final_value
            optimal_action = action
            if v == -1:
                return v, optimal_action

    return v, optimal_action
