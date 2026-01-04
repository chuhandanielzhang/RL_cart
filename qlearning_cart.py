import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
rng = np.random.default_rng(12345)

# reinforcement learning parameters ======================================
N = 100
episodes = 120
gamma = 0.99
alpha = 0.99

# physical system parameters =============================================
m = 1
dt = 0.05
force = 1

# system initialization =================================================
scope = 1 * np.pi
x_space = np.linspace(-scope, scope, N)
dx_space = np.linspace(-scope, scope, N)
[x_grid, dx_grid] = np.meshgrid(x_space, dx_space)
reward = np.zeros(np.shape(x_grid))
reward[int(N/2), int(N/2)] = 10

action_map = np.zeros((N, N))
Q = np.random.randn(N, N, 2)

fig = plt.figure(figsize=(15, 5))
nrows = 1
ncols = 3
ax = fig.add_subplot(nrows, ncols, 1)
image1 = ax.imshow(np.transpose(Q[:, :, 0]),
                   extent=[x_space[0], x_space[-1], dx_space[0], dx_space[-1]],
                   cmap='jet', origin='lower', vmin=0, vmax=10)
ax = fig.add_subplot(nrows, ncols, 2)
image2 = ax.imshow(np.transpose(Q[:, :, 1]),
                   extent=[x_space[0], x_space[-1], dx_space[0], dx_space[-1]],
                   cmap='jet', origin='lower', vmin=0, vmax=10)
ax = fig.add_subplot(nrows, ncols, 3)
image3 = ax.imshow(np.transpose(Q[:, :, 1] > Q[:, :, 0]),
                   extent=[x_space[0], x_space[-1], dx_space[0], dx_space[-1]],
                   cmap='jet', origin='lower', vmin=-0.01, vmax=0.01)
plt.tight_layout()

for episode in tqdm(range(episodes)):
    for i in range(N):
        for j in range(N):
            if i == int(N/2) and j == int(N/2):
                Q[i, j, :] = reward[i, j]
            else:
                x = x_space[i]
                dx = dx_space[j]
                a = 0
                ddx = force * (a - 0.5)*2/m
                dx += ddx * dt
                x += dx * dt
                indx_new = np.argmin(abs(x-x_space))
                inddx_new = np.argmin(abs(dx-dx_space))
                Q[i, j, a] = ((1-alpha) * Q[i, j, a]
                              + alpha * (reward[i, j] + gamma * np.max(Q[indx_new, inddx_new, :])))
                x = x_space[i]
                dx = dx_space[j]
                a = 1
                ddx = force * (a - 0.5)*2/m
                dx += ddx * dt
                x += dx * dt
                indx_new = np.argmin(abs(x-x_space))
                inddx_new = np.argmin(abs(dx-dx_space))
                Q[i, j, a] = ((1-alpha) * Q[i, j, a]
                              + alpha * (reward[i, j] + gamma * np.max(Q[indx_new, inddx_new, :])))
    image1.set_array(np.transpose(Q[:, :, 0]))
    image2.set_array(np.transpose(Q[:, :, 1]))
    image3.set_array(np.transpose(Q[:, :, 1] > Q[:, :, 0]))
    plt.title(episode)
    plt.show()
    plt.pause(1e-5)


dt_plot = 0.001
fig = plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.imshow(np.transpose(Q[:, :, 1] > Q[:, :, 0]),
                   extent=[x_space[0], x_space[-1], dx_space[0], dx_space[-1]],
                   cmap='jet', origin='lower', vmin=-1, vmax=1)
plt.tight_layout()
plt.plot(x_space, -np.sign(x_space)*np.sqrt(2) * np.sqrt(abs(force * x_space / m)), 'w--', linewidth=3)
x = 1
dx = 1
for step in tqdm(range(3500)):
    x_save = x
    dx_save = dx
    if dx > - np.sqrt(2) * np.sqrt(abs(force * x / m)):
        a = 0
    else:
        a = 1
    ddx = force * (a - 0.5)*2/m
    dx += ddx * dt_plot
    x += dx * dt_plot
    plt.subplot(1, 3, 1)
    plt.plot([x_save, x], [dx_save, dx], 'b--')
    plt.subplot(1, 3, 2)
    plt.plot([step * dt_plot, (step+1) * dt_plot], [x_save, x], 'b')
    plt.subplot(1, 3, 3)
    plt.plot([step * dt_plot, (step+1) * dt_plot], [dx_save, dx], 'b')
    threshold = 0.01
    if abs(x) < threshold and abs(dx) < threshold:
        break
plt.subplot(1, 3, 1)
plt.plot([x_save, x], [dx_save, dx], 'b', label='Optimal')
plt.subplot(1, 3, 2)
plt.plot([step * dt_plot, (step+1) * dt_plot], [x_save, x], 'b', label='Optimal')
plt.subplot(1, 3, 3)
plt.plot([step * dt_plot, (step+1) * dt_plot], [dx_save, dx], 'b', label='Optimal')
x = 1
dx = 1
A = 2
zeta = 0.1
omega = dx/A/np.cos(np.arcsin(x/A))
time_scale = 15
time = np.linspace(0, step*dt_plot*time_scale, step*time_scale)
plt.subplot(1, 3, 1)
plt.plot(np.exp(-zeta*time)*A*np.sin(omega*time+np.arcsin(x/A)),
         np.exp(-zeta*time)*A*omega*np.cos(omega*time+np.arcsin(x/A)), 'r--', label='PID')
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'$\dot{x}$ [m/s]')
plt.colorbar()
plt.legend()
plt.subplot(1, 3, 2)
plt.plot([0, step*dt_plot*time_scale], [0, 0], 'k:')
plt.plot(time, np.exp(-zeta*time)*A*np.sin(omega*time+np.arcsin(x/A)), 'r--', label='PID')
plt.xlabel(r'$Time$ [s]')
plt.ylabel(r'$x$ [m]')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot([0, step*dt_plot*time_scale], [0, 0], 'k:')
plt.plot(time, np.exp(-zeta*time)*A*omega*np.cos(omega*time+np.arcsin(x/A)), 'r--', label='PID')
plt.xlabel(r'$Time$ [s]')
plt.ylabel(r'$\dot{x}$ [m/s]')
plt.legend()
plt.tight_layout()
plt.savefig('temp.pdf')
