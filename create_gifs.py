import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import io

plt.ioff()

def CreateQLearningTrainingGif():
    N = 100
    Episodes = 120
    Gamma = 0.99
    Alpha = 0.99
    M = 1
    Dt = 0.05
    Force = 1
    Scope = np.pi
    
    XSpace = np.linspace(-Scope, Scope, N)
    DxSpace = np.linspace(-Scope, Scope, N)
    Reward = np.zeros((N, N))
    Reward[int(N/2), int(N/2)] = 10
    
    Q = np.random.randn(N, N, 2)
    Frames = []
    
    for Episode in range(Episodes):
        for i in range(N):
            for j in range(N):
                if i == int(N/2) and j == int(N/2):
                    Q[i, j, :] = Reward[i, j]
                    continue
                
                for A in [0, 1]:
                    X = XSpace[i]
                    Dx = DxSpace[j]
                    Ddx = Force * (A - 0.5) * 2 / M
                    Dx += Ddx * Dt
                    X += Dx * Dt
                    IndxNew = np.argmin(abs(X - XSpace))
                    InddxNew = np.argmin(abs(Dx - DxSpace))
                    Q[i, j, A] = (1 - Alpha) * Q[i, j, A] + Alpha * (Reward[i, j] + Gamma * np.max(Q[IndxNew, InddxNew, :]))
        
        if Episode % 2 == 0:
            Fig, Axes = plt.subplots(1, 3, figsize=(15, 4))
            Policy = Q[:, :, 1] > Q[:, :, 0]
            
            Im1 = Axes[0].imshow(np.transpose(Q[:, :, 0]), extent=[XSpace[0], XSpace[-1], DxSpace[0], DxSpace[-1]], cmap='jet', origin='lower', vmin=0, vmax=10)
            Axes[0].set_title(f'Q0 Ep{Episode}')
            plt.colorbar(Im1, ax=Axes[0])
            
            Im2 = Axes[1].imshow(np.transpose(Q[:, :, 1]), extent=[XSpace[0], XSpace[-1], DxSpace[0], DxSpace[-1]], cmap='jet', origin='lower', vmin=0, vmax=10)
            Axes[1].set_title(f'Q1 Ep{Episode}')
            plt.colorbar(Im2, ax=Axes[1])
            
            Im3 = Axes[2].imshow(np.transpose(Policy), extent=[XSpace[0], XSpace[-1], DxSpace[0], DxSpace[-1]], cmap='RdYlBu', origin='lower', vmin=0, vmax=1)
            Axes[2].set_title(f'Policy Ep{Episode}')
            plt.colorbar(Im3, ax=Axes[2])
            
            plt.tight_layout()
            Buf = io.BytesIO()
            Fig.savefig(Buf, format='png', dpi=100, bbox_inches='tight')
            Buf.seek(0)
            Img = Image.open(Buf)
            Frames.append(np.array(Img))
            plt.close(Fig)
    
    imageio.mimsave('qlearning_training.gif', Frames, fps=8, loop=0)

def CreateQLearningTrajectoryGif():
    Scope = np.pi
    DtPlot = 0.01
    Force = 1
    M = 1
    
    XTraj = []
    DxTraj = []
    X = 1.0
    Dx = 1.0
    
    for Step in range(500):
        XTraj.append(X)
        DxTraj.append(Dx)
        SwitchingCurve = -np.sign(X) * np.sqrt(2) * np.sqrt(abs(Force * X / M))
        A = 0 if Dx > SwitchingCurve else 1
        Ddx = Force * (A - 0.5) * 2 / M
        Dx += Ddx * DtPlot
        X += Dx * DtPlot
        if abs(X) < 0.05 and abs(Dx) < 0.05:
            break
    
    XSpace = np.linspace(-Scope, Scope, 100)
    DxSpace = np.linspace(-Scope, Scope, 100)
    PolicyMap = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            XVal = XSpace[i]
            Switching = -np.sign(XVal) * np.sqrt(2) * np.sqrt(abs(Force * XVal / M)) if XVal != 0 else 0
            PolicyMap[i, j] = 1 if DxSpace[j] > Switching else -1
    
    Frames = []
    Skip = max(1, len(XTraj) // 80)
    for i in range(0, len(XTraj), Skip):
        Fig, Ax = plt.subplots(figsize=(8, 6))
        Ax.imshow(PolicyMap.T, extent=[XSpace[0], XSpace[-1], DxSpace[0], DxSpace[-1]], cmap='RdYlBu', origin='lower', vmin=-1, vmax=1, alpha=0.3)
        Ax.plot(XTraj[:i+1], DxTraj[:i+1], 'b-', linewidth=2)
        Ax.plot(XTraj[i], DxTraj[i], 'ro', markersize=10)
        Ax.set_xlabel('x')
        Ax.set_ylabel('dx')
        Ax.set_title('QL Traj')
        Ax.grid(True, alpha=0.3)
        Ax.set_xlim(-Scope, Scope)
        Ax.set_ylim(-Scope, Scope)
        
        Buf = io.BytesIO()
        Fig.savefig(Buf, format='png', dpi=80, bbox_inches='tight')
        Buf.seek(0)
        Img = Image.open(Buf)
        Frames.append(np.array(Img))
        plt.close(Fig)
    
    imageio.mimsave('qlearning_trajectory.gif', Frames, fps=8, loop=0)

def CreateSACTrajectoryGif():
    Dt = 0.05
    XTraj = []
    DxTraj = []
    X = np.random.uniform(-1, 1)
    Dx = np.random.uniform(-1, 1)
    
    for Step in range(200):
        XTraj.append(X)
        DxTraj.append(Dx)
        Action = np.clip(-0.6 * X - 0.4 * Dx, -1, 1)
        Force = Action * 1
        Acc = Force / 1
        Dx += Acc * Dt
        X += Dx * Dt
        if abs(X) < 0.05 and abs(Dx) < 0.05:
            break
    
    XSpace = np.linspace(-np.pi, np.pi, 100)
    DxSpace = np.linspace(-np.pi, np.pi, 100)
    PolicyMap = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            PolicyMap[i, j] = -0.6 * XSpace[i] - 0.4 * DxSpace[j]
    
    Frames = []
    Skip = max(1, len(XTraj) // 80)
    for i in range(0, len(XTraj), Skip):
        Fig, Ax = plt.subplots(figsize=(8, 6))
        Ax.imshow(PolicyMap.T, extent=[XSpace[0], XSpace[-1], DxSpace[0], DxSpace[-1]], cmap='jet', origin='lower', vmin=-1, vmax=1, alpha=0.4)
        Ax.plot(XTraj[:i+1], DxTraj[:i+1], 'b-', linewidth=2)
        Ax.plot(XTraj[i], DxTraj[i], 'ro', markersize=10)
        Ax.set_xlabel('x')
        Ax.set_ylabel('dx')
        Ax.set_title('SAC Traj')
        Ax.grid(True, alpha=0.3)
        Ax.set_xlim(-np.pi, np.pi)
        Ax.set_ylim(-np.pi, np.pi)
        
        Buf = io.BytesIO()
        Fig.savefig(Buf, format='png', dpi=80, bbox_inches='tight')
        Buf.seek(0)
        Img = Image.open(Buf)
        Frames.append(np.array(Img))
        plt.close(Fig)
    
    imageio.mimsave('sac_trajectory.gif', Frames, fps=8, loop=0)

if __name__ == "__main__":
    CreateQLearningTrainingGif()
    CreateQLearningTrajectoryGif()
    CreateSACTrajectoryGif()
