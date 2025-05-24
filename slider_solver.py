"""
Name = Sovit Bhandari
Description = This code implements an A* search algorithm to solve the nxn sliding puzzle by finding the shortest sequence of moves to reach the 
goal state. It uses the Manhattan distance heuristic and explores neighboring states efficiently, ensuring optimality. The function returns a 
list of tile moves to solve the puzzle.
"""

from typing import List

def solveSlider(size: int, grid: List[int]) -> List[int]:
    """
    Given the board as a flat list in row-major order, the puzzle is solved when the list is exactly
    [0, 1, 2, ..., (size*size)-1]. Note that 0 stands for the empty space and must always be at the
    very beginning (top-left corner). This function returns a list of moves, where each move is the tile
    number that slides into the empty space, that will solve the puzzle.
    """
    
    start = tuple(grid)                                            					 # Convert the input list to a tuple so we can use it as a key in dictionaries.
    goal = tuple(range(size * size))      											# The goal state is simply the numbers in order with 0 at the start.
    
    # If we're already at the goal, no moves are needed.
    if start == goal:
        return []
    
    def heuristic(state: tuple) -> int:
        """
        Calculate the total Manhattan distance for all tiles except the empty one.
        For each tile, the target position is determined by its value: For tile v (v != 0), its goal position is (v // size, v % size).
        This gives us an admissible estimate of the moves needed.
        """
        total = 0
        for idx, tile in enumerate(state):
            if tile != 0:
                current_row, current_col = divmod(idx, size)
                goal_row, goal_col = divmod(tile, size)
                total += abs(current_row - goal_row) + abs(current_col - goal_col)
        return total

    def get_neighbors(state: tuple) -> List[tuple]:
        """
        Generate all the neighbor states from the current state by sliding a tile into the empty space.
        Neighbors are produced by moving the tile above, below, to the left, or to the right of the empty spot.
        It Returns a list of tuples where each tuple is (neighbor_state, moved_tile) with moved_tile being the
        number of the tile that was moved.
        """
        
        neighbors = []
        empty_index = state.index(0)
        row, col = divmod(empty_index, size)
        
        # List possible moves (up, down, left, right) if they are within bounds.
        adjacent_positions = []
        if row > 0:
            adjacent_positions.append((row - 1, col))
        if row < size - 1:
            adjacent_positions.append((row + 1, col))
        if col > 0:
            adjacent_positions.append((row, col - 1))
        if col < size - 1:
            adjacent_positions.append((row, col + 1))
        
        for new_row, new_col in adjacent_positions:
            neighbor_index = new_row * size + new_col
            moved_tile = state[neighbor_index]   											 # Identify which tile is moving into the empty space.
            state_list = list(state)														 # Create a new state by swapping the empty tile with the adjacent tile.
            state_list[empty_index], state_list[neighbor_index] = state_list[neighbor_index], state_list[empty_index]
            neighbors.append((tuple(state_list), moved_tile))
        
        return neighbors

    # A* search implementation:
    # Each entry in open_set is a tuple: (priority, counter, state, moves)
    # 'priority' = cost so far + heuristic estimate.
    open_set = []
    counter = 0  																			 # Used to break ties when priorities are equal.
    start_cost = 0
    start_priority = start_cost + heuristic(start)
    open_set.append((start_priority, counter, start, []))
    
    best_cost = {start: 0}     															   # This dict stores the smallest cost (g-value) found so far for each state.
    
    while open_set:
        # We sort the open_set each time because we're not allowed to import any other function.
        open_set.sort(key=lambda x: x[0])
        current_priority, _, current_state, path = open_set.pop(0)
        current_cost = best_cost[current_state]
        
        # Check if we have reached the goal state.
        if current_state == goal:
            return path
        
        # Process each neighbor of the current state.
        for neighbor, moved_tile in get_neighbors(current_state):
            new_cost = current_cost + 1  													# Every move costs 1.
            if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                best_cost[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor)
                counter += 1
                open_set.append((priority, counter, neighbor, path + [moved_tile]))
    
    # If no solution is found, it return an empty move list.
    return []
