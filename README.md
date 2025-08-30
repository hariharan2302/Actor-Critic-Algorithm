# 🎮 Actor-Critic Reinforcement Learning Implementation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-red.svg)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28+-green.svg)](https://gymnasium.farama.org)

A comprehensive implementation of the **Actor-Critic (A2C)** algorithm applied to multiple reinforcement learning environments, from simple grid worlds to complex continuous control and image-based tasks.

## 📋 Table of Contents
- [Overview](#overview)
- [Environments](#environments)
- [Architecture](#architecture)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Key Features](#key-features)

## 🎯 Overview

This project implements the **Advantage Actor-Critic (A2C)** algorithm, a policy gradient method that combines the benefits of both value-based and policy-based reinforcement learning approaches. The implementation demonstrates the versatility of A2C across various environment types:

- **Discrete environments**: GridWorld, CartPole
- **Continuous control**: MountainCar, BipedalWalker  
- **Image-based environments**: CarRacing

### 🧠 What is Actor-Critic?

The Actor-Critic method consists of two neural networks:
- **Actor (Policy Network)**: Learns the optimal policy π(a|s) - what action to take in each state
- **Critic (Value Network)**: Learns the value function V(s) - estimates the expected return from each state

The critic provides feedback to the actor, reducing variance in policy gradient estimates and improving learning stability.

## 🌍 Environments

### 1. 🎲 Custom GridWorld Environment
- **State Space**: 4×4 grid (16 discrete states)
- **Action Space**: 4 discrete actions (up, down, left, right)
- **Objective**: Navigate from start (0,0) to goal (3,3) while avoiding obstacles
- **Rewards**: Goal (+10), Obstacles (-1, -2, -5)

### 2. 🎪 CartPole-v1
- **State Space**: 4 continuous variables (position, velocity, angle, angular velocity)
- **Action Space**: 2 discrete actions (left, right)
- **Objective**: Balance pole on cart for 500 timesteps
- **Success Criteria**: Average reward ≥ 500

### 3. 🏔️ MountainCarContinuous-v0
- **State Space**: 2 continuous variables (position, velocity)
- **Action Space**: 1 continuous action (force applied)
- **Objective**: Drive car up the mountain to reach the flag
- **Challenge**: Sparse rewards, requires momentum building

### 4. 🚶 BipedalWalker-v3
- **State Space**: 24 continuous variables (hull angle, angular velocity, leg positions, etc.)
- **Action Space**: 4 continuous actions (hip/knee joint torques)
- **Objective**: Walk forward as far as possible without falling
- **Complexity**: High-dimensional continuous control

### 5. 🏎️ CarRacing-v2
- **State Space**: 96×96×3 RGB images
- **Action Space**: 3 continuous actions (steering, gas, brake)
- **Objective**: Complete racing track as quickly as possible
- **Challenge**: Vision-based control with CNN feature extraction

## 🏗️ Architecture

### Standard A2C Agent (Discrete/Continuous)
```
Input Layer → Hidden Layer (16 units) → Output Layer
                    ↓
Actor Network: State → Action Probabilities/Values
Critic Network: State → State Value
```

### Image-based A2C Agent (CarRacing)
```
CNN Feature Extractor:
96×96×3 → Conv2D → Conv2D → Flatten → FC Layers

Actor: Features → Action Distribution
Critic: Features → State Value
```

## 🔧 Implementation Details

### Key Components

1. **Neural Network Architecture**
   - Fully connected networks with ReLU activation
   - Separate networks for actor and critic
   - CNN-based feature extraction for image inputs

2. **Training Algorithm**
   - Policy gradient with baseline (advantage estimation)
   - Discounted reward calculation
   - Adam optimizer for both networks
   - MSE loss for critic, policy gradient loss for actor

3. **Hyperparameters**
   | Environment | Episodes | Learning Rate | Discount Factor |
   |-------------|----------|---------------|-----------------|
   | GridWorld   | 1000     | 0.001         | 0.99           |
   | CartPole    | 1500     | 0.004         | 0.99           |
   | MountainCar | 1500     | 0.0005        | 0.8            |
   | BipedalWalker| 15000   | 0.0001        | 0.99           |
   | CarRacing   | 200      | 0.0005        | 0.99           |

## 📊 Results

### Training Performance

| Environment | Success Criteria | Episodes to Solve | Final Performance |
|-------------|------------------|-------------------|-------------------|
| GridWorld   | Reach goal       | ~200              | ✅ Solved         |
| CartPole    | Avg reward ≥ 500 | ~700              | ✅ Solved         |
| MountainCar | Avg reward ≥ -100| ~200              | ✅ Solved         |
| BipedalWalker| Stable walking  | ~1000+            | 🔄 Training       |
| CarRacing   | Complete track   | ~200+             | 🔄 Training       |

### Learning Curves
The implementation includes visualization of training progress with reward plots showing:
- Episode vs. Average Reward
- Convergence patterns
- Performance stability

## 🚀 Installation

### Prerequisites
```bash
pip install torch torchvision
pip install gymnasium[all]
pip install numpy matplotlib
pip install pickle
pip install PIL
```

### Clone Repository
```bash
git clone <repository-url>
cd Actor-Critic-Algorithm
```

## 💻 Usage

### Running the Complete Implementation
```python
# Open the Jupyter notebook
jupyter notebook hvenkatr_sugheerth_assignment3_part1_part2.ipynb
```

### Loading Trained Models
```python
import pickle
import torch

# Load trained models
with open('sugheert_hvenkatr_assignment3_cartpole_actor.pkl', 'rb') as f:
    actor_model = pickle.load(f)

with open('sugheert_hvenkatr_assignment3_cartpole_critic.pkl', 'rb') as f:
    critic_model = pickle.load(f)
```

### Training New Agent
```python
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1")

# Initialize agent
agent = A2C_Agent(env, max_reward=500, episodes=1500, 
                  discount=0.99, lr_actor=0.004, lr_critic=0.004)

# Train the agent
agent.train()

# Test the agent
agent.test()
```

## 📁 File Structure

```
├── hvenkatr_sugheerth_assignment3_part1_part2.ipynb    # Main implementation
├── README.md                                            # This file
├── LICENSE                                             # License file
│
├── Trained Models:
├── sugheert_hvenkatr_assignment3_gridworld_actor.pkl   # GridWorld actor
├── sugheert_hvenkatr_assignment3_gridworld_critic.pkl  # GridWorld critic
├── sugheert_hvenkatr_assignment3_cartpole_actor.pkl    # CartPole actor
├── sugheert_hvenkatr_assignment3_cartpole_critic.pkl   # CartPole critic
├── sugheert_hvenkatr_assignment3_carracing_actor.pkl   # CarRacing actor
├── sugheert_hvenkatr_assignment3_carracing_critic.pkl  # CarRacing critic
├── sugheert_hvenkatr_assignment3_bipedal_walker_actor.pkl  # BipedalWalker actor
├── sugheert_hvenkatr_assignment3_bipedal_walker_critic.pkl # BipedalWalker critic
└── hvenkatr_sugheert_assignment3_part2_MountainCarContinuous (1).pkl # MountainCar weights
```

## ✨ Key Features

### 🎯 Multi-Environment Support
- **Discrete Action Spaces**: GridWorld, CartPole
- **Continuous Action Spaces**: MountainCar, BipedalWalker, CarRacing
- **Image-based Observations**: CarRacing with CNN processing

### 🧪 Advanced Techniques
- **Advantage Estimation**: Reduces variance in policy gradients
- **Separate Optimizers**: Independent learning rates for actor and critic
- **Experience Replay**: Efficient learning from collected trajectories
- **Reward Normalization**: Improved training stability

### 📈 Comprehensive Analysis
- **Training Visualization**: Real-time learning curves
- **Performance Metrics**: Episode rewards, convergence analysis
- **Model Persistence**: Save/load trained models
- **Testing Framework**: Evaluation on trained agents

### 🔧 Modular Design
- **Reusable Components**: Base classes for different environment types
- **Configurable Hyperparameters**: Easy experimentation
- **Clean Architecture**: Separate concerns for different functionalities

## 🎓 Educational Value

This implementation serves as an excellent learning resource for:
- Understanding policy gradient methods
- Implementing actor-critic algorithms
- Working with different types of RL environments
- Handling both discrete and continuous action spaces
- Processing image-based observations with CNNs

---

**This project is for educational purposes as part of CSE 446/546 coursework.**