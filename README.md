# Leak Detection Bot Simulation

Leak Detection Bot Simulation is a project that simulates a leak detection scenario within a grid-based ship layout. The simulation features multiple bots, each employing different detection algorithms to find and plug leaks. The primary goal is to compare the efficiency and effectiveness of these bots in navigating the ship and handling leaks.

The project dynamically generates a grid representing the ship, with random obstacles and open paths, ensuring each simulation run is unique. Leaks are randomly placed within the grid, requiring bots to locate and plug them. Various detection algorithms are implemented, including heuristic-based movement, Breadth-First Search (BFS), probabilistic sensing, and combined probabilistic and heuristic approaches. Additionally, the project features bots capable of multi-leak detection.

The bots implemented include:

Bot1: Utilizes a heuristic approach with a fixed detection radius to find leaks.
Bot2: Navigates the grid using BFS and detects leaks in the vicinity.
Bot3: Implements a probabilistic model to sense and locate leaks based on city block distance.
Bot4: Combines probabilistic sensing with a listening period before moving towards potential leak locations.
Bot5: Uses a mix of heuristic movement and BFS for leak detection and plugging.
Bot6: Detects all leaks within a vicinity and uses BFS for precise location and plugging.
Bot7: Employs combined probabilistic sensing and BFS to navigate towards high-probability leak areas.
Bot8: Enhances probabilistic sensing by combining probabilities from multiple leaks.
Bot9: Integrates probabilistic sensing, listening duration, and BFS for efficient leak handling.
Leak detection is a critical task in many industrial and maritime contexts. Efficient detection and plugging of leaks can prevent significant damage and loss. This project serves as a simulation framework to explore and compare different strategies for autonomous leak detection, which can inform real-world applications and improvements in robotic systems.

## Table of Contents
- [Overview](#overview)
- [Grid Layout](#grid-layout)
- [Bots](#bots)
- [Usage](#usage)
- [Example Output](#example-output)
- [Contributing](#contributing)
- [License](#license)

## Overview

The simulation generates a grid-based ship layout with randomly placed leaks. Bots navigate through the ship to detect and plug these leaks using various algorithms. The project compares the performance of these bots in terms of the number of actions taken to plug all leaks.

## Grid Layout

The ship is represented as a 2D grid of size `grid_size x grid_size`. Each cell in the grid can be:
- `blocked`: Represents an obstacle or a wall.
- `open`: Represents an open space that the bot can move through.
- `leak`: Represents a leak that needs to be plugged.

The grid is initialized with all cells blocked except for a random interior square that is opened. Cells are iteratively opened to form a navigable layout with potential dead ends.

## Bots

### Bot1
Uses a simple heuristic to move towards open cells and detects leaks in its vicinity using a fixed detection radius.

### Bot2
Uses a Breadth-First Search (BFS) algorithm to navigate the grid and detect leaks in its vicinity.

### Bot3
Implements a probabilistic approach to sense leaks and update leak probabilities using city block distance.

### Bot4
Combines probabilistic sensing with a listening duration to decide when to move towards potential leak locations.

### Bot5
Uses a combination of heuristic movement and BFS to detect and plug leaks.

### Bot6
Detects all leaks in its vicinity and uses BFS to narrow down and plug leaks.

### Bot7
Combines probabilistic sensing with BFS pathfinding to move towards high probability leak locations.

### Bot8
Enhances Bot7 by combining probabilities from multiple leaks to decide the next move.

### Bot9
Uses probabilistic sensing with a listening duration and BFS to find and plug leaks efficiently.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/leak-detection-bot.git
   cd leak-detection-bot
