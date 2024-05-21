from collections import deque
import random
import math
import numpy as np
from scipy.spatial.distance import cityblock
from queue import Queue

class Ship:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = [['blocked' for _ in range(grid_size)] for _ in range(grid_size)]
        self.bot_position = None
        self.leak_position = None
        self.leak_positions = []
        
       

    def generate_ship(self):
        # Initialize the ship's grid
        self.grid = [['blocked' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Open a random square in the interior
        interior_square = (random.randint(1, self.grid_size - 2), random.randint(1, self.grid_size - 2))
        self.grid[interior_square[0]][interior_square[1]] = 'open'

        # Continue opening cells iteratively
        while True:
            cells_to_open = []
            for i in range(1, self.grid_size - 1):
                for j in range(1, self.grid_size - 1):
                    if self.grid[i][j] == 'blocked':
                        neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
                        open_neighbors = sum(1 for x, y in neighbors if self.grid[x][y] == 'open')
                        if open_neighbors == 1:
                            cells_to_open.append((i, j))
            
            if not cells_to_open:
                break
            
            cell_to_open = random.choice(cells_to_open)
            self.grid[cell_to_open[0]][cell_to_open[1]] = 'open'
        
        # Identify dead ends and open half of them
        dead_ends = []
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                if self.grid[i][j] == 'open':
                    neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
                    open_neighbors = sum(1 for x, y in neighbors if self.grid[x][y] == 'open')
                    if open_neighbors == 1:
                        dead_ends.append((i, j))
        
        random.shuffle(dead_ends)
        for i in range(len(dead_ends) // 2):
            x, y = dead_ends[i]
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            closed_neighbors = [(x, y) for x, y in neighbors if self.grid[x][y] == 'blocked']
            if closed_neighbors:
                self.grid[closed_neighbors[0][0]][closed_neighbors[0][1]] = 'open'
        
        open_cells = [(i, j) for i in range(1, self.grid_size - 1) for j in range(1, self.grid_size - 1) if self.grid[i][j] == 'open']
        self.bot_position = random.choice(open_cells)
        open_cells.remove(self.bot_position)
        self.generate_leaks(1)  # Generate one leak initially

    def generate_leaks(self, number_of_leaks=2):
        # Ensure there's enough space for the number of leaks requested
        open_cells = [(i, j) for i in range(1, self.grid_size - 1) for j in range(1, self.grid_size - 1) if self.grid[i][j] == 'open']
        assert len(open_cells) >= number_of_leaks, "Not enough open cells to place leaks"
        
        # Clear any existing leaks before generating new ones
        for x, y in self.leak_positions:
            self.grid[x][y] = 'open'
        self.leak_positions.clear()
        
        # Randomly place leaks
        for _ in range(number_of_leaks):
            leak_position = random.choice(open_cells)
            open_cells.remove(leak_position)
            self.leak_positions.append(leak_position)
            x, y = leak_position
            self.grid[x][y] = 'leak'
        
    def plug_leak(self, leak_position):
        if leak_position in self.leak_positions:
            self.leak_positions.remove(leak_position)
            x, y = leak_position
            self.grid[x][y] = 'open'
            print(f"Leak at {leak_position} has been plugged.")
        else:
            print("No leak at the specified position to plug.")

    def print_ship(self):
        for row in self.grid:
            print(" ".join(row))

    def is_valid_move(self, position):
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[x][y] == 'open'

    def move_bot(self, next_position):
        if self.is_valid_move(next_position):
            self.bot_position = next_position
            return True
        return False
    
    def get_leak_locations(self):
        return self.leak_positions
    
    def get_bot_location(self):
        return self.bot_position

class Bot1:
    def __init__(self, ship, detection_radius):
        self.ship = ship
        self.detection_radius = int(detection_radius)
        self.detected_leak = False
        self.no_leak = set()
        self.num_actions = 0

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def detect_leak_in_vicinity(self):
        x, y = self.ship.get_bot_location()
        leak_found = False
        self.num_actions += 1
        for i in range(-self.detection_radius, self.detection_radius + 1):
            for j in range(-self.detection_radius, self.detection_radius + 1):
                if 0 <= x+i < self.ship.grid_size and 0 <= y+j < self.ship.grid_size:
                    if self.ship.grid[x+i][y+j] == 'leak':
                        leak_found = True
                        self.detected_leak = True
                        return True
        # Mark all cells in the detection square as 'no_leak' if no leak is detected
        if not leak_found:
            for i in range(-self.detection_radius, self.detection_radius + 1):
                for j in range(-self.detection_radius, self.detection_radius + 1):
                    if 0 <= x+i < self.ship.grid_size and 0 <= y+j < self.ship.grid_size:
                        if self.ship.grid[x+i][y+j] == 'open':
                            self.ship.grid[x+i][y+j] = 'no_leak'
        return False

    def cells_in_detection_square(self):
        x, y = self.ship.get_bot_location()
        cells = []
        for i in range(-self.detection_radius, self.detection_radius + 1):
            for j in range(-self.detection_radius, self.detection_radius + 1):
                if 0 <= x+i < self.ship.grid_size and 0 <= y+j < self.ship.grid_size:
                    cells.append((x+i, y+j))
        return cells

    def bfs_search_for_leak(self, start):
        queue = deque([start])
        visited = set([start])
        
        while queue:
            x, y = queue.popleft()
            self.num_actions += 1

            
            print(f"Bot checking cell: ({x}, {y})")

            if self.ship.grid[x][y] == 'leak':
                print(f"Bot found the leak at ({x}, {y})!")
                return (x, y)
            
            # Possible directions: up, down, left, right
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.ship.grid_size and 0 <= ny < self.ship.grid_size and
                    self.ship.grid[nx][ny] not in ['blocked', 'no_leak'] and
                    (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        
        return None


    def move(self):
        current_location = self.ship.get_bot_location()
        x, y = current_location

        # Check immediate neighbors for an open cell to move to
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.ship.grid_size and 0 <= ny < self.ship.grid_size and
                self.ship.grid[nx][ny] == 'open' and
                (nx, ny) not in self.no_leak):
                self.no_leak.add((nx, ny))
                self.ship.move_bot((nx, ny))
                self.num_actions += 1
                if self.num_actions > 10000:
                    print("Action limit exceeded (10000). Stopping bot.")
                    return  # Stop executing further actions
                print(f"Bot moved one step to {self.ship.get_bot_location()}")
                return False

        # If no open cell is found nearby, proceed with the detection or BFS
        if self.detected_leak:
            next_move = self.bfs_search_for_leak(current_location)
            if next_move:
                self.ship.move_bot(next_move)
                print(f"Bot is currently at {next_move} and found the leak!")
                print("Leak Plugged!")
                return True  # Indicates the task is completed
        else:
            if self.detect_leak_in_vicinity():
                print(f"Bot is currently at {current_location} and detected a leak in its vicinity!")
                return False  # Continue searching
            else:
                print("No open cell nearby and no leak in vicinity.")
                print(f"Total number of actions taken: {self.num_actions}")
                return True  # Indicates no further action is possible


    def plug_leak(self):
        while True:
            if self.num_actions > 10000:
                print("Action limit exceeded (10000). Unable to plug the leak.")
                break  
            task_completed = self.move()
            if task_completed:
                break

        
        print(f"Total number of actions taken: {self.num_actions}")


class Bot2:
    def __init__(self, ship, detection_radius):
        self.ship = ship
        self.detection_radius = int(detection_radius)
        self.detected_leak = False
        self.visited_cells = set()
        self.num_actions = 0
        

    def detect_leak_in_vicinity(self, position):
        x, y = position
        self.num_actions += 1
        for i in range(-self.detection_radius, self.detection_radius + 1):
            for j in range(-self.detection_radius, self.detection_radius + 1):
                if 0 <= x+i < self.ship.grid_size and 0 <= y+j < self.ship.grid_size:
                    if self.ship.grid[x+i][y+j] == 'leak':
                        return True
        return False

    def bfs_search(self):
        queue = deque([self.ship.get_bot_location()])
        visited = set([self.ship.get_bot_location()])

        while queue:
            current_position = queue.popleft()
            x, y = current_position

            self.ship.move_bot(current_position)

            print(f"Bot2 is currently at ({x}, {y}).")
            self.num_actions +=1
            if self.num_actions > 10000:
                print("Action limit exceeded (10000). Stopping bot.")
                return  # Stop executing further actions
            if self.detect_leak_in_vicinity(current_position):
                print(f"Bot2 detected a leak near ({x}, {y})!")
                self.detected_leak = True
                return current_position  # Return the cell where the leak was detected

            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                next_position = (nx, ny)
                if self.ship.is_valid_move(next_position) and next_position not in visited:
                    visited.add(next_position)
                    queue.append(next_position)

        return None  # If no leak is detected, return None

    def bfs_search_for_leak(self, start):
        queue = deque([start])
        visited = set([start])
        
        while queue:
            current_position = queue.popleft()
            self.num_actions +=1
            x, y = current_position
            self.ship.move_bot(current_position)

            print(f"Bot2 is narrowing its search and checking cell: ({x}, {y})")
            self.num_actions +=1

            if self.ship.grid[x][y] == 'leak':
                print(f"Bot2 found the leak at ({x}, {y})!")
                print(f"Leak plugged.")
                print(f"Total number of actions taken: {self.num_actions}")
                return (x, y)
            
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.ship.grid_size and 0 <= ny < self.ship.grid_size and
                    self.ship.grid[nx][ny] not in ['blocked', 'no_leak'] and
                    (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return None  

    def plug_leak(self):
        if self.num_actions > 10000:
            print("Action limit exceeded (10000). Unable to plug the leak.")
            return  # Stop the loop if action limit is reached
        leak_detected_pos = self.bfs_search()
        if leak_detected_pos:
            # Now, use bfs_search_for_leak to find the exact position of the leak
            self.bfs_search_for_leak(leak_detected_pos)
        else:
            print("Bot2 couldn't detect the leak!")
            print(f"Total number of actions taken: {self.num_actions}")

class Bot3:
    def __init__(self, ship, alpha):
        self.ship = ship
        self.prob_map = np.full_like(ship.grid, 1/(ship.grid_size - 1), dtype=float)
        self.position = ship.get_bot_location()
        self.prob_map[self.position] = 0
        self.alpha = alpha  # This can be adjusted based on how quickly the probability decays with distance
        self.num_actions = 0

    def sense(self):
        beep_probabilities = []
        self.num_actions += 1 
        for leak_position in self.ship.get_leak_locations():
            d = cityblock(self.position, leak_position)
            if d == 0:
                return True  # Bot is at the leak
            prob_beep = np.exp(-self.alpha * (d - 1))
            beep_probabilities.append(prob_beep)

    def update_probabilities(self, sensed_beep):
        for (x, y), _ in np.ndenumerate(self.prob_map):
            if self.ship.grid[x][y] != 'blocked':  # Consider only non-blocked cells
                d = cityblock(self.position, (x, y))
                prob_beep = np.exp(-self.alpha * (d - 1)) if d > 0 else 1
                if sensed_beep:
                    self.prob_map[x, y] *= prob_beep
                else:
                    self.prob_map[x, y] *= (1 - prob_beep)
        
        # Normalize probabilities
        total_prob = np.sum(self.prob_map)
        if total_prob > 0:
            self.prob_map /= total_prob

    def choose_next_cell(self):
        # Find the cell with the highest probability that is not blocked
        max_prob = 0
        candidates = []
        for (x, y), prob in np.ndenumerate(self.prob_map):
            if self.ship.grid[x][y] != 'blocked' and prob > max_prob:
                candidates = [(x, y)]
                max_prob = prob
            elif self.ship.grid[x][y] != 'blocked' and prob == max_prob:
                candidates.append((x, y))

        # If there are no 'open' cells, something went wrong
        if not candidates:
            raise ValueError("No available 'open' cells to move to.")

        # Break ties by distance, then choose randomly if needed
        min_distance = np.inf
        best_candidate = None
        for candidate in candidates:
            distance = cityblock(self.position, candidate)
            if distance < min_distance:
                min_distance = distance
                best_candidate = candidate
            elif distance == min_distance and np.random.rand() < 0.5:
                best_candidate = candidate

        return tuple(best_candidate)

    def bfs_to_target(self, start, target):
        """ Perform BFS to find the shortest path from start to target """
        if start == target:
            return [start]  # If already at the target

        queue = deque([start])
        visited = {start}
        parent_map = {start: None}

        found_target = False
        while queue and not found_target:
            current = queue.popleft()
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
                next_cell = (current[0] + direction[0], current[1] + direction[1])
                if next_cell == target:
                    parent_map[next_cell] = current
                    found_target = True
                    break
                if next_cell not in visited and self.ship.is_valid_move(next_cell):
                    queue.append(next_cell)
                    visited.add(next_cell)
                    parent_map[next_cell] = current

        if not found_target:
            return []  # Return an empty path if target is not reachable

        path = []
        while target is not None:
            path.append(target)
            target = parent_map.get(target)  

        path.reverse()  # Reverse the path to start from the bot's current position
        return path

    def move_to_target(self, target):
        """ Move to target using BFS pathfinding """
        path = self.bfs_to_target(self.position, target)
        # Move along the path (excluding the starting cell, which is the current position)
        for next_cell in path[1:]:
            self.position = next_cell
            self.prob_map[self.position] = 0  # Set the visited cell's probability to 0
            print(f"Moving to {self.position}, Leak Probabilities Updated.")
            self.num_actions += 1

    def move(self):
        if self.num_actions > 10000:
            print("Action limit exceeded (10000). Stopping bot.")
            return  # Stop executing further actions
        sensed_beep = self.sense()
        self.update_probabilities(sensed_beep)
        next_cell = self.choose_next_cell()
        self.move_to_target(next_cell)  # Move along the BFS path

    def plug_leak(self):
        while self.ship.leak_positions:  # While there are leaks left
            if self.num_actions > 10000:
                print("Action limit exceeded (10000). Unable to plug the leak.")
                break  # Stop the loop if action limit is reached
            self.move()
            print(f"Moving to {self.position}, Leak Probabilities Updated.")
            # Since there can be multiple leaks, check if the bot's current position is in any of the leaks
            if self.position in self.ship.get_leak_locations():
                print("Leak plugged.")
                print(f"Total number of actions taken: {self.num_actions}")
                self.ship.plug_leak(self.position)  
                break

class Bot4:
    def __init__(self, ship, alpha, listen_duration):
        self.ship = ship
        self.prob_map = np.full_like(ship.grid, 1/(ship.grid_size - 2)**2, dtype=float)
        self.position = ship.get_bot_location()
        self.prob_map[self.position] = 0
        self.alpha = alpha  # Decay rate for the probability map
        self.listen_duration = listen_duration  # Number of turns to listen before moving
        self.listen_counter = 0  # Counter for the listening turns
        self.num_actions = 0
        self.leak_position = ship.leak_positions[0] if ship.leak_positions else None


    def sense(self):
        self.num_actions+=1
        d = cityblock(self.position, self.leak_position)
        if d == 0:
            return True  # Bot is at the leak
        prob_beep = np.exp(-self.alpha * (d - 1))
        return np.random.rand() < prob_beep

    def update_probabilities(self, sensed_beep):
        for (x, y), _ in np.ndenumerate(self.prob_map):
            if self.ship.grid[x][y] != 'blocked':  # Consider only non-blocked cells
                d = cityblock(self.position, (x, y))
                prob_beep = np.exp(-self.alpha * (d - 1)) if d > 0 else 1
                if sensed_beep:
                    self.prob_map[x, y] *= prob_beep
                else:
                    self.prob_map[x, y] *= (1 - prob_beep)
        
        # Normalize probabilities
        total_prob = np.sum(self.prob_map)
        if total_prob > 0:
            self.prob_map /= total_prob

    def choose_next_cell(self):
        # Find the cell with the highest probability that is not blocked
        max_prob = 0
        candidates = []
        for (x, y), prob in np.ndenumerate(self.prob_map):
            if self.ship.grid[x][y] != 'blocked' and prob > max_prob:
                candidates = [(x, y)]
                max_prob = prob
            elif self.ship.grid[x][y] != 'blocked' and prob == max_prob:
                candidates.append((x, y))

        # If there are no 'open' cells, something went wrong
        if not candidates:
            raise ValueError("No available 'open' cells to move to.")

        # Break ties by distance, then choose randomly if needed
        min_distance = np.inf
        best_candidate = None
        for candidate in candidates:
            distance = cityblock(self.position, candidate)
            if distance < min_distance:
                min_distance = distance
                best_candidate = candidate
            elif distance == min_distance and np.random.rand() < 0.5:
                best_candidate = candidate

        return tuple(best_candidate)

    def bfs_to_target(self, start, target):
        """ Perform BFS to find the shortest path from start to target """
        if start == target:
            return [start]  # If already at the target

        queue = deque([start])
        visited = {start}
        parent_map = {start: None}

        found_target = False
        while queue and not found_target:
            current = queue.popleft()
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
                next_cell = (current[0] + direction[0], current[1] + direction[1])
                if next_cell == target:
                    parent_map[next_cell] = current
                    found_target = True
                    break
                if next_cell not in visited and self.ship.is_valid_move(next_cell):
                    queue.append(next_cell)
                    visited.add(next_cell)
                    parent_map[next_cell] = current

        if not found_target:
            return []  # Return an empty path if target is not reachable

        path = []
        while target is not None:
            path.append(target)
            target = parent_map.get(target)  # Use get to avoid KeyError

        path.reverse()  # Reverse the path to start from the bot's current position
        return path

    def move_to_target(self, target):
        """ Move to target using BFS pathfinding """
        path = self.bfs_to_target(self.position, target)
        # Move along the path (excluding the starting cell, which is the current position)
        for next_cell in path[1:]:
            self.position = next_cell
            self.prob_map[self.position] = 0  # Set the visited cell's probability to 0
            print(f"Moving to {self.position}, Leak Probabilities Updated.")
            self.num_actions+=1

    def move(self):
        if self.num_actions > 10000:
            print("Action limit exceeded (10000). Stopping bot.")
            return  # Stop executing further actions
        
        if self.listen_counter < self.listen_duration:
            # Listen without moving
            sensed_beep = self.sense()
            self.update_probabilities(sensed_beep)
            self.listen_counter += 1
            print(f"Listening at {self.position}, Leak Probabilities Updated.")
        else:
            # Reset listen counter and move towards the target
            self.listen_counter = 0
            sensed_beep = self.sense()
            self.update_probabilities(sensed_beep)
            next_cell = self.choose_next_cell()
            self.move_to_target(next_cell)  # Move along the BFS path to next cell
            print(f"Moving to {self.position}, Leak Probabilities Updated.")
            self.num_actions+=1

    def plug_leak(self):
         while self.position != self.leak_position:
            if self.num_actions > 10000:
                print("Action limit exceeded (10000). Unable to plug the leak.")
                break  # Stop the loop if action limit is reached

            self.move()
            if self.position == self.leak_position:
                print("Leak plugged at location:", self.leak_position)
                print(f"Total number of actions taken: {self.num_actions}")
                break
            else:
                # Output to see what the bot is doing (if not at leak location)
                print(f"Current Position: {self.position}, Leak Probabilities Updated.")



class Bot5:
    def __init__(self, ship, detection_radius):
        self.ship = ship
        self.detection_radius = int(detection_radius)
        self.detected_leak = False
        self.no_leak = set()
        self.position = ship.get_bot_location()
        self.num_actions = 0

    def _distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def detect_leak_in_vicinity(self):
        self.num_actions += 1 
        x, y = self.ship.get_bot_location()
        leak_found = False
        for i in range(-self.detection_radius, self.detection_radius + 1):
            for j in range(-self.detection_radius, self.detection_radius + 1):
                if 0 <= x+i < self.ship.grid_size and 0 <= y+j < self.ship.grid_size:
                    if self.ship.grid[x+i][y+j] == 'leak':
                        leak_found = True
                        self.detected_leak = True
                        return True
        # Mark all cells in the detection square as 'no_leak' if no leak is detected
        if not leak_found:
            for i in range(-self.detection_radius, self.detection_radius + 1):
                for j in range(-self.detection_radius, self.detection_radius + 1):
                    if 0 <= x+i < self.ship.grid_size and 0 <= y+j < self.ship.grid_size:
                        if self.ship.grid[x+i][y+j] == 'open':
                            self.ship.grid[x+i][y+j] = 'no_leak'
        return False

    def cells_in_detection_square(self):
        x, y = self.ship.get_bot_location()
        cells = []
        for i in range(-self.detection_radius, self.detection_radius + 1):
            for j in range(-self.detection_radius, self.detection_radius + 1):
                if 0 <= x+i < self.ship.grid_size and 0 <= y+j < self.ship.grid_size:
                    cells.append((x+i, y+j))
        return cells

    def bfs_search_for_leak(self, start):
        queue = deque([start])
        visited = set([start])
        
        while queue:
            current_position = queue.popleft()
            x, y = current_position
            self.ship.move_bot(current_position)

            # Report every move of the bot
            print(f"Bot checking cell: ({x}, {y})")
            self.num_actions +=1

            if self.ship.grid[x][y] == 'leak':
                print(f"Bot found the leak at ({x}, {y})!")
                return (x, y)
            
            # Possible directions: up, down, left, right
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.ship.grid_size and 0 <= ny < self.ship.grid_size and
                    self.ship.grid[nx][ny] not in ['blocked', 'no_leak'] and
                    (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        # This should ideally never be reached if the leak is within the vicinity
        return None


    def move(self):
        current_location = self.ship.get_bot_location()
        x, y = current_location

        # Check immediate neighbors for an open cell to move to
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.ship.grid_size and 0 <= ny < self.ship.grid_size and
                self.ship.grid[nx][ny] == 'open' and
                (nx, ny) not in self.no_leak):
                self.no_leak.add((nx, ny))
                self.ship.move_bot((nx, ny))
                self.position = self.ship.get_bot_location()  # Update the bot's position after moving
                print(f"Bot moved one step to {self.position}")
                self.num_actions+=1
                if self.num_actions > 10000:
                    print("Action limit exceeded (10000). Stopping bot.")
                    return  # Stop executing further actions
                return False

        # If no open cell is found nearby, proceed with the detection or BFS
        if self.detected_leak:
            next_move = self.bfs_search_for_leak(current_location)
            if next_move:
                self.ship.move_bot(next_move)
                self.position = next_move  # Update the bot's position after moving
                print(f"Bot is currently at {next_move} and found the leak!")
                if self.position in self.ship.get_leak_locations():
                    self.ship.plug_leak(self.position)
                self.detected_leak = False
                self.no_leak.clear()  
                if not self.ship.leak_positions:
                    print("All leaks have been successfully plugged.")
                    return True  # All leaks are plugged, bot can stop
                else:
                    return False  # There are still leaks, bot should continue
        else:
            if self.detect_leak_in_vicinity():
                print(f"Bot is currently at {current_location} and detected a leak in its vicinity!")
                print(f"Total number of actions taken: {self.num_actions}")
            else:
                print("No open cell nearby and no leak in vicinity.")
                print(f"Total number of actions taken: {self.num_actions}")
                return True  # Return True to indicate that the bot could not find a new path and may need to end the loop
        return False

    def plug_leak(self):
        while self.ship.leak_positions:  
            if self.num_actions > 10000:
                print("Action limit exceeded (10000). Unable to plug the leak.")
                break  # Stop the loop if action limit is reached
            if self.move():  # If the move method returns True, it means no further action can be taken (no path or leaks detected)
                break
            # Check if the bot's current position is a leak and plug it
            if self.position in self.ship.get_leak_locations():
                self.ship.plug_leak(self.position)  
                print("Leak plugged.")


class Bot6:
    def __init__(self, ship, detection_radius):
        self.ship = ship
        self.detection_radius = int(detection_radius)
        self.detected_leak = False
        self.visited_cells = set()
        self.position = ship.get_bot_location()
        self.num_actions = 0

    def detect_all_leaks_in_vicinity(self, position):
        self.num_actions+=1
        detected_leaks = []
        x, y = position
        for i in range(-self.detection_radius, self.detection_radius + 1):
            for j in range(-self.detection_radius, self.detection_radius + 1):
                if 0 <= x+i < self.ship.grid_size and 0 <= y+j < self.ship.grid_size:
                    if self.ship.grid[x+i][y+j] == 'leak':
                        detected_leaks.append((x+i, y+j))
        return detected_leaks

    def bfs_search(self):
        queue = deque([self.ship.get_bot_location()])
        visited = set([self.ship.get_bot_location()])

        leaks_found = []

        while queue:
            current_position = queue.popleft()
            x, y = current_position
            self.ship.move_bot(current_position)
            print(f"Bot6 is currently checking ({x}, {y}).")
            self.num_actions+=1
            if self.num_actions > 10000:
                print("Action limit exceeded (10000). Stopping bot.")
                return  # Stop executing further actions


            leaks_in_vicinity = self.detect_all_leaks_in_vicinity((x, y))
            if leaks_in_vicinity:
                leaks_found.extend(leaks_in_vicinity)
                        
            if not leaks_found:
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (self.ship.is_valid_move((nx, ny)) and (nx, ny) not in visited):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            else:
                for leak in leaks_found:
                    self.bfs_search_for_leak(leak)
                leaks_found = []
                if not self.ship.leak_positions:
                    return True

        if not leaks_found:
            print("Bot6 has finished checking and found no leaks.")
            return False  # No leaks were found

    def bfs_search_for_leak(self, start):
        queue = deque([start])
        visited = set([start])
        
        while queue:
            current_position = queue.popleft()
            x, y = current_position
            self.ship.move_bot(current_position)
            self.num_actions+=1

            print(f"Bot6 is narrowing its search and checking cell: ({x}, {y})")

            if self.ship.grid[x][y] == 'leak':
                print(f"Bot6 found the leak at ({x}, {y})!")
                print(f"Total number of actions taken: {self.num_actions}")
                self.ship.plug_leak(current_position)
                self.detected_leak = False
                if not self.ship.leak_positions:
                    return
                return (x, y)
            
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.ship.grid_size and 0 <= ny < self.ship.grid_size and
                    self.ship.grid[nx][ny] not in ['blocked', 'no_leak'] and
                    (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return None  
    def plug_leak(self):
        while self.ship.leak_positions:
            if self.num_actions > 10000:
                print("Action limit exceeded (10000). Unable to plug the leak.")
                break
            leak_detected = self.bfs_search()
            if not leak_detected:
                print("No leaks detected, Bot6 is stopping.")
                break  # Exit the loop if no leaks were detected

            # If leaks were detected, continue with the leak plugging process
            for leak_position in self.ship.leak_positions:
                self.bfs_search_for_leak(leak_position)
                if not self.ship.leak_positions:
                    break  # Break the loop if all leaks have been plugged

class Bot7:
    def __init__(self, ship, alpha):
        self.ship = ship
        self.prob_map = np.full_like(ship.grid, 1/(ship.grid_size - 1), dtype=float)
        self.position = ship.get_bot_location()
        self.prob_map[self.position] = 0
        self.alpha = alpha  # This can be adjusted based on how quickly the probability decays with distance
        self.num_actions = 0

    def sense(self):
        self.num_actions+=1
        beep_probabilities = []
        for leak_position in self.ship.get_leak_locations():
            d = cityblock(self.position, leak_position)
            if d == 0:
                return True  # Bot is at the leak
            prob_beep = np.exp(-self.alpha * (d - 1))
            beep_probabilities.append(prob_beep)

        # If the bot can hear beeps from multiple leaks, combine probabilities
        combined_prob = 1 - np.prod([1 - p for p in beep_probabilities])
        return np.random.rand() < combined_prob

    def update_probabilities(self, sensed_beep):
        for (x, y), _ in np.ndenumerate(self.prob_map):
            if self.ship.grid[x][y] != 'blocked':  # Consider only non-blocked cells
                d = cityblock(self.position, (x, y))
                prob_beep = np.exp(-self.alpha * (d - 1)) if d > 0 else 1
                if sensed_beep:
                    self.prob_map[x, y] *= prob_beep
                else:
                    self.prob_map[x, y] *= (1 - prob_beep)
        
        # Normalize probabilities
        total_prob = np.sum(self.prob_map)
        if total_prob > 0:
            self.prob_map /= total_prob

    def choose_next_cell(self):
        # Find the cell with the highest probability that is not blocked
        max_prob = 0
        candidates = []
        for (x, y), prob in np.ndenumerate(self.prob_map):
            if self.ship.grid[x][y] != 'blocked' and prob > max_prob:
                candidates = [(x, y)]
                max_prob = prob
            elif self.ship.grid[x][y] != 'blocked' and prob == max_prob:
                candidates.append((x, y))

        # If there are no 'open' cells, something went wrong
        if not candidates:
            raise ValueError("No available 'open' cells to move to.")

        # Break ties by distance, then choose randomly if needed
        min_distance = np.inf
        best_candidate = None
        for candidate in candidates:
            distance = cityblock(self.position, candidate)
            if distance < min_distance:
                min_distance = distance
                best_candidate = candidate
            elif distance == min_distance and np.random.rand() < 0.5:
                best_candidate = candidate

        return tuple(best_candidate)

    def bfs_to_target(self, start, target):
        """ Perform BFS to find the shortest path from start to target """
        if start == target:
            return [start]  # If already at the target

        queue = deque([start])
        visited = {start}
        parent_map = {start: None}

        found_target = False
        while queue and not found_target:
            current = queue.popleft()
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
                next_cell = (current[0] + direction[0], current[1] + direction[1])
                if next_cell == target:
                    parent_map[next_cell] = current
                    found_target = True
                    break
                if next_cell not in visited and self.ship.is_valid_move(next_cell):
                    queue.append(next_cell)
                    visited.add(next_cell)
                    parent_map[next_cell] = current

        if not found_target:
            return []  # Return an empty path if target is not reachable

        path = []
        while target is not None:
            path.append(target)
            target = parent_map.get(target)  

        path.reverse()  # Reverse the path to start from the bot's current position
        return path

    def move_to_target(self, target):
        """ Move to target using BFS pathfinding """
        path = self.bfs_to_target(self.position, target)
        # Move along the path (excluding the starting cell, which is the current position)
        for next_cell in path[1:]:
            self.position = next_cell
            self.prob_map[self.position] = 0  # Set the visited cell's probability to 0
            self.num_actions+=1
            print(f"Moving to {self.position}, Leak Probabilities Updated.")

    def move(self):
        if self.num_actions > 10000:
            print("Action limit exceeded (10000). Stopping bot.")
            return  # Stop executing further actions
        sensed_beep = self.sense()
        self.update_probabilities(sensed_beep)
        next_cell = self.choose_next_cell()
        self.move_to_target(next_cell)  # Move along the BFS path

    def plug_leak(self):
        while self.ship.leak_positions:  
            if self.num_actions > 10000:
                print("Action limit exceeded (10000). Unable to plug the leak.")
                break
            self.move()
            # Since there can be multiple leaks, check if the bot's current position is in any of the leaks
            while self.position in self.ship.get_leak_locations():
                print("Leak plugged.")
                print(f"Total number of actions taken: {self.num_actions}")
                self.ship.plug_leak(self.position)  

class Bot8:
    def __init__(self, ship, alpha):
        self.ship = ship
        self.prob_map = np.full_like(ship.grid, 1/(ship.grid_size - 1), dtype=float)
        self.position = ship.get_bot_location()
        self.prob_map[self.position] = 0
        self.alpha = alpha  
        self.num_actions = 0

    def sense(self):
        self.num_actions+=1
        beep_probabilities = []
        for leak_position in self.ship.get_leak_locations():
            d = cityblock(self.position, leak_position)
            if d == 0:
                return True  # Bot is at the leak
            prob_beep = np.exp(-self.alpha * (d - 1))
            beep_probabilities.append(prob_beep)

        # If the bot can hear beeps from multiple leaks, combine probabilities
        combined_prob = 1 - np.prod([1 - p for p in beep_probabilities])
        return np.random.rand() < combined_prob

    def update_probabilities(self, sensed_beep):
        # Probabilities if beep is heard or not from each leak
        beep_probabilities = np.zeros_like(self.prob_map)
        no_beep_probabilities = np.zeros_like(self.prob_map)

        # Calculate probabilities for each cell from each leak
        for leak_position in self.ship.get_leak_locations():
            for (x, y), _ in np.ndenumerate(self.prob_map):
                if self.ship.grid[x][y] != 'blocked':  # Consider only non-blocked cells
                    d = cityblock(self.position, (x, y))
                    prob_beep = np.exp(-self.alpha * (d - 1)) if d > 0 else 1
                    beep_probabilities[x, y] += prob_beep
                    no_beep_probabilities[x, y] += (1 - prob_beep)

        # Normalize the probabilities from multiple leaks to a 0-1 range
        beep_probabilities = np.clip(beep_probabilities, 0, 1)
        no_beep_probabilities = np.clip(no_beep_probabilities, 0, 1)

        # Update the probability map based on whether a beep was sensed
        if sensed_beep:
            self.prob_map *= beep_probabilities
        else:
            # Calculate the combined probability of not hearing a beep from any leaks
            combined_no_beep_prob = 1 - (1 - no_beep_probabilities)**len(self.ship.get_leak_locations())
            self.prob_map *= combined_no_beep_prob

        # Normalize probabilities
        total_prob = np.sum(self.prob_map)
        if total_prob > 0:
            self.prob_map /= total_prob

    def choose_next_cell(self):
        # Find the cell with the highest probability that is not blocked
        max_prob = 0
        candidates = []
        for (x, y), prob in np.ndenumerate(self.prob_map):
            if self.ship.grid[x][y] != 'blocked' and prob > max_prob:
                candidates = [(x, y)]
                max_prob = prob
            elif self.ship.grid[x][y] != 'blocked' and prob == max_prob:
                candidates.append((x, y))

        # If there are no 'open' cells, something went wrong
        if not candidates:
            raise ValueError("No available 'open' cells to move to.")

        # Break ties by distance, then choose randomly if needed
        min_distance = np.inf
        best_candidate = None
        for candidate in candidates:
            distance = cityblock(self.position, candidate)
            if distance < min_distance:
                min_distance = distance
                best_candidate = candidate
            elif distance == min_distance and np.random.rand() < 0.5:
                best_candidate = candidate

        return tuple(best_candidate)

    def bfs_to_target(self, start, target):
        """ Perform BFS to find the shortest path from start to target """
        if start == target:
            return [start]  # If already at the target

        queue = deque([start])
        visited = {start}
        parent_map = {start: None}

        found_target = False
        while queue and not found_target:
            current = queue.popleft()
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
                next_cell = (current[0] + direction[0], current[1] + direction[1])
                if next_cell == target:
                    parent_map[next_cell] = current
                    found_target = True
                    break
                if next_cell not in visited and self.ship.is_valid_move(next_cell):
                    queue.append(next_cell)
                    visited.add(next_cell)
                    parent_map[next_cell] = current

        if not found_target:
            return []  # Return an empty path if target is not reachable

        path = []
        while target is not None:
            path.append(target)
            target = parent_map.get(target)  

        path.reverse()  # Reverse the path to start from the bot's current position
        return path

    def move_to_target(self, target):
        """ Move to target using BFS pathfinding """
        path = self.bfs_to_target(self.position, target)
        # Move along the path (excluding the starting cell, which is the current position)
        for next_cell in path[1:]:
            self.position = next_cell
            self.prob_map[self.position] = 0  # Set the visited cell's probability to 0
            self.num_actions+=1
            print(f"Moving to {self.position}, Leak Probabilities Updated.")

    def move(self):
        if self.num_actions > 10000:
            print("Action limit exceeded (10000). Stopping bot.")
            return  # Stop executing further actions
        sensed_beep = self.sense()
        self.update_probabilities(sensed_beep)
        next_cell = self.choose_next_cell()
        self.move_to_target(next_cell)  # Move along the BFS path

    def plug_leak(self):
        while self.ship.leak_positions:  
            self.move()
            if self.num_actions > 10000:
                print("Action limit exceeded (10000). Unable to plug the leak.")
                break  # Stop the loop if action limit is reached
            # Since there can be multiple leaks, check if the bot's current position is in any of the leaks
            while self.position in self.ship.get_leak_locations():
                print("Leak plugged.")
                print(f"Total number of actions taken: {self.num_actions}")
                self.ship.plug_leak(self.position)  

class Bot9:
    def __init__(self, ship, alpha, listen_duration):
        self.ship = ship
        self.prob_map = np.full_like(ship.grid, 1/(ship.grid_size - 2)**2, dtype=float)
        self.position = ship.get_bot_location()
        self.prob_map[self.position] = 0
        self.alpha = alpha
        self.listen_duration = listen_duration
        self.listen_counter = 0
        self.num_actions = 0

    def sense(self):
        self.num_actions+=1
        beep_probabilities = []
        for leak_position in self.ship.get_leak_locations():
            d = cityblock(self.position, leak_position)
            if d == 0:
                return True  # Bot is at one of the leaks
            prob_beep = np.exp(-self.alpha * (d - 1))
            beep_probabilities.append(prob_beep)

        combined_prob = 1 - np.prod([1 - p for p in beep_probabilities])
        return np.random.rand() < combined_prob

    def update_probabilities(self, sensed_beep):
        beep_probabilities = np.zeros_like(self.prob_map)
        no_beep_probabilities = np.zeros_like(self.prob_map)

        for leak_position in self.ship.get_leak_locations():
            for (x, y), _ in np.ndenumerate(self.prob_map):
                if self.ship.grid[x][y] != 'blocked':
                    d = cityblock(self.position, (x, y))
                    prob_beep = np.exp(-self.alpha * (d - 1)) if d > 0 else 1
                    beep_probabilities[x, y] += prob_beep
                    no_beep_probabilities[x, y] += (1 - prob_beep)

        beep_probabilities = np.clip(beep_probabilities, 0, 1)
        no_beep_probabilities = np.clip(no_beep_probabilities, 0, 1)

        if sensed_beep:
            self.prob_map *= beep_probabilities
        else:
            combined_no_beep_prob = 1 - (1 - no_beep_probabilities)**len(self.ship.get_leak_locations())
            self.prob_map *= combined_no_beep_prob

        total_prob = np.sum(self.prob_map)
        if total_prob > 0:
            self.prob_map /= total_prob

    def choose_next_cell(self):
        non_blocked_probs = np.ma.masked_where(self.ship.grid == 'blocked', self.prob_map)
        max_prob = np.max(non_blocked_probs)
        candidates = list(zip(*np.where(non_blocked_probs == max_prob)))
        min_distance = np.inf
        best_candidate = None
        for candidate in candidates:
            distance = cityblock(self.position, candidate)
            if distance < min_distance:
                min_distance = distance
                best_candidate = candidate
            elif distance == min_distance and random.random() < 0.5:
                best_candidate = candidate
        return best_candidate

    def bfs_to_target(self, start, target):
        """ Perform BFS to find the shortest path from start to target """
        if start == target:
            return [start]  # If already at the target

        queue = deque([start])
        visited = {start}
        parent_map = {start: None}

        found_target = False
        while queue and not found_target:
            current = queue.popleft()
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
                next_cell = (current[0] + direction[0], current[1] + direction[1])
                if next_cell == target:
                    parent_map[next_cell] = current
                    found_target = True
                    break
                if next_cell not in visited and self.ship.is_valid_move(next_cell):
                    queue.append(next_cell)
                    visited.add(next_cell)
                    parent_map[next_cell] = current

        if not found_target:
            return []  # Return an empty path if target is not reachable

        path = []
        while target is not None:
            path.append(target)
            target = parent_map.get(target)  

        path.reverse()  # Reverse the path to start from the bot's current position
        return path

    def move_to_target(self, target):
        """ Move to target using BFS pathfinding """
        path = self.bfs_to_target(self.position, target)
        # Move along the path (excluding the starting cell, which is the current position)
        for next_cell in path[1:]:
            self.position = next_cell
            self.prob_map[self.position] = 0  # Set the visited cell's probability to 0
            self.num_actions+=1
            print(f"Moving to {self.position}, Leak Probabilities Updated.")

    def move(self):
        if self.num_actions > 10000:
            print("Action limit exceeded (10000). Stopping bot.")
            return  # Stop executing further actions
        if self.listen_counter < self.listen_duration:
            sensed_beep = self.sense()
            self.update_probabilities(sensed_beep)
            self.listen_counter += 1
            print(f"Listening at {self.position}, Leak Probabilities Updated.")
        else:
            self.listen_counter = 0
            next_cell = self.choose_next_cell()
            self.move_to_target(next_cell)

    def plug_leak(self):
        while any(self.ship.get_leak_locations()):
            self.move()
            if self.num_actions > 10000:
                print("Action limit exceeded (10000). Unable to plug the leak.")
                break  # Stop the loop if action limit is reached
            while self.position in self.ship.get_leak_locations():
                print("Leak plugged at location:", self.position)
                print(f"Total number of actions taken: {self.num_actions}")
                self.ship.plug_leak(self.position)
                # After plugging a leak, we should reset the probabilities
                self.prob_map = np.full_like(self.ship.grid, 1/(self.ship.grid_size - 2)**2, dtype=float)
                self.prob_map[self.position] = 0


def main():
   
    '''
    grid_size = 50  # Adjust grid size as needed
    total_actions = 0
    successful_runs = 0
    k = 1  # Detection range for Bot1

    for _ in range(8000):
        ship = Ship(grid_size)
        ship.generate_ship()
        ship.generate_leaks(2)

        bot5 = Bot5(ship, k)
        bot5.plug_leak()

        if ship.leak_positions:  # Check if there are still leaks left
            # If there are leaks left, it means the bot failed to plug at least one leak
            continue

        # If all leaks are plugged, count this as a successful run and add the actions
        successful_runs += 1
        total_actions += bot5.num_actions

    if successful_runs > 0:
        average_actions = total_actions / successful_runs   
        print(f"Average number of actions taken over {successful_runs} successful runs: {average_actions}")
    else:
        print("No successful runs were recorded.")
    '''
    grid_size = 50  
    ship = Ship(grid_size)
    ship.generate_ship()
    ship.generate_leaks(2)
    total_actions = 0
    '''
    for _ in range(50):
        ship = Ship(grid_size)
        ship.generate_ship()
        #ship.generate_leaks(2)

        k = 0.25  
        alpha = 5
        listen_duration = 1
        bot6 = Bot6(ship,alpha,listen_duration)

        bot6.plug_leak()
        total_actions += bot6.num_actions

    average_actions = total_actions / 50
    print(f"Average number of actions taken over 100 runs: {average_actions}")
    '''
    
    k = 1  
    alpha = 1
    listen_duration = 3

    bot1 = Bot1(ship,k)
    bot2 = Bot2(ship,k)
    bot3 = Bot3(ship,alpha)
    bot4 = Bot4(ship,alpha,listen_duration)
    bot5 = Bot5(ship, k)
    bot6 = Bot6(ship,k)
    bot7 = Bot7(ship,alpha)
    bot8 = Bot8(ship,alpha)
    bot9 = Bot9(ship,alpha,listen_duration)

    print("Initial ship layout:")
    ship.print_ship()
    print("Leak Location:")
    print(ship.get_leak_locations())
    print("bot initial location:")
    print(ship.get_bot_location())
    bot1.plug_leak()
    bot2.plug_leak()
    bot3.plug_leak()
    bot4.plug_leak()
    bot5.plug_leak()
    bot6.plug_leak()
    bot7.plug_leak()
    bot8.plug_leak()
    bot9.plug_leak()
   

if __name__ == "__main__":
    main()
