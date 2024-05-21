# Leak Detection Bot Simulation

This project simulates a leak detection scenario within a grid-based ship layout. Multiple bots with different detection algorithms are implemented to find and plug leaks in the ship. The project aims to compare the efficiency and effectiveness of different bot algorithms.

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
