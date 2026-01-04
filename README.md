# RL Cart Control

Two RL algorithms for cart control: Q-Learning (discrete) and SAC (continuous).

## Stack

Q-Learning: numpy, matplotlib  
SAC: PyTorch, numpy, matplotlib  
GIF: imageio, PIL

## Q-Learning

Discrete state-action space, Q-table iteration. 100×100 grid, actions {0,1}.  
Params: α=0.99, γ=0.99, 120 episodes

**Training:**
<img src="qlearning_training.gif" width="800">

**Trajectory:**
<img src="qlearning_trajectory.gif" width="600">

## SAC

Continuous control, Actor-Critic. State 2D, action [-1,1].  
Networks: Actor 2→128→1, Twin Critic 3→128→1  
Params: lr=5e-4, γ=0.99, batch=64, 20 episodes

**Trajectory:**
<img src="sac_trajectory.gif" width="600">

## Run

```bash
python qlearning_cart.py
python sac_cart.py
python create_gifs.py
```
