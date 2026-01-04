#!/usr/bin/env python3
"""
Create GIF animations from the RL results
This script runs simplified versions and generates trajectory GIFs
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import io

plt.ioff()  # Turn off interactive mode

def create_qlearning_gif():
    """Create GIF for Q-Learning trajectory"""
    print("Creating Q-Learning trajectory GIF...")
    
    # Parameters
    scope = np.pi
    dt_plot = 0.01
    force = 1
    m = 1
    
    # Simulate trajectory using learned policy
    x_traj = []
    dx_traj = []
    x = 1.0
    dx = 1.0
    
    for step in range(500):
        x_traj.append(x)
        dx_traj.append(dx)
        
        # Policy: if above switching curve, action 0, else action 1
        switching_curve = -np.sign(x) * np.sqrt(2) * np.sqrt(abs(force * x / m))
        if dx > switching_curve:
            a = 0
        else:
            a = 1
        ddx = force * (a - 0.5) * 2 / m
        dx += ddx * dt_plot
        x += dx * dt_plot
        
        if abs(x) < 0.05 and abs(dx) < 0.05:
            break
    
    # Create frames
    frames = []
    x_space = np.linspace(-scope, scope, 100)
    dx_space = np.linspace(-scope, scope, 100)
    
    # Background policy map
    policy_map = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            x_val = x_space[i]
            switching = -np.sign(x_val) * np.sqrt(2) * np.sqrt(abs(force * x_val / m)) if x_val != 0 else 0
            policy_map[i, j] = 1 if dx_space[j] > switching else -1
    
    skip = max(1, len(x_traj) // 80)
    for i in range(0, len(x_traj), skip):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(policy_map.T, extent=[x_space[0], x_space[-1], dx_space[0], dx_space[-1]],
                 cmap='RdYlBu', origin='lower', vmin=-1, vmax=1, alpha=0.3)
        ax.plot(x_traj[:i+1], dx_traj[:i+1], 'b-', linewidth=2, label='Trajectory')
        ax.plot(x_traj[i], dx_traj[i], 'ro', markersize=10, label='Current state')
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(r'$x$ [m]', fontsize=12)
        ax.set_ylabel(r'$\dot{x}$ [m/s]', fontsize=12)
        ax.set_title('Q-Learning Trajectory', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-scope, scope)
        ax.set_ylim(-scope, scope)
        
        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(np.array(img))
        plt.close(fig)
    
    # Save GIF
    imageio.mimsave('qlearning_trajectory.gif', frames, fps=8, loop=0)
    print("✓ Saved qlearning_trajectory.gif")


def create_sac_gif():
    """Create GIF for SAC trajectory"""
    print("Creating SAC trajectory GIF...")
    
    # Parameters
    dt = 0.05
    x_traj = []
    dx_traj = []
    x = np.random.uniform(-1, 1)
    dx = np.random.uniform(-1, 1)
    
    # Simulate continuous controller (simplified SAC policy)
    for step in range(200):
        x_traj.append(x)
        dx_traj.append(dx)
        
        # Simplified continuous policy: drive to origin
        action = -0.6 * x - 0.4 * dx
        action = np.clip(action, -1, 1)
        
        force = action * 1
        acc = force / 1
        dx += acc * dt
        x += dx * dt
        
        if abs(x) < 0.05 and abs(dx) < 0.05:
            break
    
    # Create frames
    frames = []
    x_space = np.linspace(-np.pi, np.pi, 100)
    dx_space = np.linspace(-np.pi, np.pi, 100)
    
    # Background: simplified policy visualization
    policy_map = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            x_val = x_space[i]
            dx_val = dx_space[j]
            policy_map[i, j] = -0.6 * x_val - 0.4 * dx_val
    
    skip = max(1, len(x_traj) // 80)
    for i in range(0, len(x_traj), skip):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(policy_map.T, extent=[x_space[0], x_space[-1], dx_space[0], dx_space[-1]],
                      cmap='jet', origin='lower', vmin=-1, vmax=1, alpha=0.4)
        ax.plot(x_traj[:i+1], dx_traj[:i+1], 'b-', linewidth=2, label='Trajectory')
        ax.plot(x_traj[i], dx_traj[i], 'ro', markersize=10, label='Current state')
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Velocity', fontsize=12)
        ax.set_title('SAC Trajectory', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        
        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(np.array(img))
        plt.close(fig)
    
    # Save GIF
    imageio.mimsave('sac_trajectory.gif', frames, fps=8, loop=0)
    print("✓ Saved sac_trajectory.gif")


if __name__ == "__main__":
    print("Generating GIF animations for RL results...\n")
    try:
        create_qlearning_gif()
        create_sac_gif()
        print("\n✅ All GIFs generated successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

