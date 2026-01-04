import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)

# Reinforcement learning parameters ======================================
plot_every = 1
N = 100
Max_step = 500
Episodes = 20
gamma = 0.99
alpha = 0.2  # 0.99
batch_size = 64
learning_rate_actor = 5e-4  # 3
learning_rate_critic = 5e-4
entropy_target = -1.0

# Physical system parameters =============================================
m = 1
dt = 0.05
force = 1


def soft_update(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def step(state_, action_):
    pos, vel = state_
    applied_force = action_[0] * force
    acc = applied_force / m
    vel += acc * dt
    pos += vel * dt
    new_state = np.array([pos, vel])
    reward_ = -np.square(pos) - np.square(vel)
    done_ = np.abs(pos) < 0.01 and np.abs(vel) < 0.01
    return new_state, reward_, done_


def build_actor():
    return nn.Sequential(nn.Linear(2, 128), nn.ReLU(),
                         nn.Linear(128, 1), nn.Tanh())


def build_critic():
    return nn.Sequential(nn.Linear(3, 128), nn.ReLU(),
                         nn.Linear(128, 1))


actor = build_actor()
critic1 = build_critic()
critic2 = build_critic()
target_critic1 = build_critic()
target_critic2 = build_critic()

actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate_actor)
critic1_optimizer = optim.Adam(critic1.parameters(), lr=learning_rate_critic)
critic2_optimizer = optim.Adam(critic2.parameters(), lr=learning_rate_critic)
log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
alpha_optimizer = optim.Adam([log_alpha], lr=learning_rate_actor)

soft_update(target_critic1, critic1, tau=1.0)
soft_update(target_critic2, critic2, tau=1.0)

# Training ==========================================
loop = []
rewards = []
fig = plt.figure(figsize=(15, 5))
nrows = 1
ncols = 3
x_space = np.linspace(-np.pi, np.pi, N)
dx_space = np.linspace(-np.pi, np.pi, N)
[X, dX] = np.meshgrid(x_space, dx_space)

for episode in range(Episodes):
    state = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
    episode_reward = 0
    pos_his = []
    vel_his = []
    for step_count in range(Max_step):
        action = actor(torch.FloatTensor(state).unsqueeze(0)).detach().cpu().numpy().flatten()
        [next_state, reward, done] = step(state, action)
        loop.append((state, action, reward, next_state, done))
        state = next_state
        episode_reward += reward
        pos_his.append(state[0])
        vel_his.append(state[1])

        if len(loop) > batch_size:
            batch = np.random.choice(len(loop), batch_size, replace=False)
            batch_data = [loop[i] for i in batch]
            [states, actions, rewards_batch, next_states, dones] = zip(*batch_data)
            states = torch.FloatTensor(np.array(states))
            actions = torch.FloatTensor(np.array(actions))
            rewards_tensor = torch.FloatTensor(np.array(rewards_batch)).unsqueeze(1)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

            with torch.no_grad():
                next_actions = actor(next_states)
                next_q1 = target_critic1(torch.cat([next_states, next_actions], dim=1))
                next_q2 = target_critic2(torch.cat([next_states, next_actions], dim=1))
                next_q = torch.min(next_q1, next_q2) - log_alpha.exp() * next_actions
                target_q = rewards_tensor + (1 - dones) * gamma * next_q

            q1 = critic1(torch.cat([states, actions], dim=1))
            q2 = critic2(torch.cat([states, actions], dim=1))

            critic1_loss = nn.MSELoss()(q1, target_q)
            critic2_loss = nn.MSELoss()(q2, target_q)
            critic1_optimizer.zero_grad()
            critic1_loss.backward()
            nn.utils.clip_grad_norm_(critic1.parameters(), max_norm=0.5)
            critic1_optimizer.step()
            critic2_optimizer.zero_grad()
            critic2_loss.backward()
            nn.utils.clip_grad_norm_(critic2.parameters(), max_norm=0.5)
            critic2_optimizer.step()

            pi = actor(states)
            q1_pi = critic1(torch.cat([states, pi], dim=1))
            q2_pi = critic2(torch.cat([states, pi], dim=1))
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (log_alpha.exp() * pi - min_q_pi).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            actor_optimizer.step()

            alpha_loss = -(log_alpha * (pi + entropy_target).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            soft_update(target_critic1, critic1)
            soft_update(target_critic2, critic2)

        if done:
            break

    rewards.append(episode_reward)
    print(f'Episode {episode}, Total Reward: {episode_reward}')

    if episode % plot_every == 0:
        policy_mean = np.zeros_like(X)
        q_values = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([X[i, j], dX[i, j]])
                action = actor(torch.FloatTensor(state).unsqueeze(0)).detach().cpu().numpy().flatten()
                policy_mean[i, j] = action[0]
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_tensor = torch.FloatTensor(action).unsqueeze(0)
                q_values[i, j] = critic1(torch.cat([state_tensor, action_tensor], dim=1)).item()

        plt.subplot(nrows, ncols, 1)
        plt.imshow(np.transpose(policy_mean), extent=[x_space[0], x_space[-1], dx_space[0], dx_space[-1]],
                   cmap='jet', origin='lower', vmin=-1, vmax=1)
        plt.title('Policy Mean (Action)')
        plt.xlabel('Position')
        plt.ylabel('Velocity')

        plt.subplot(nrows, ncols, 2)
        plt.imshow(np.transpose(q_values), extent=[x_space[0], x_space[-1], dx_space[0], dx_space[-1]],
                   cmap='jet', origin='lower', vmin=q_values.min(), vmax=q_values.max())
        plt.title('Q-Value Estimates')
        plt.xlabel('Position')
        plt.ylabel('Velocity')

        plt.subplot(nrows, ncols, 3)
        plt.plot(rewards, 'b-*')
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.tight_layout()
        plt.show()
        plt.pause(0.001)

plt.subplot(nrows, ncols, 1)
state = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
for step_count in range(Max_step):
    action = actor(torch.FloatTensor(state).unsqueeze(0)).detach().cpu().numpy().flatten()
    [next_state, reward, done] = step(state, action)
    state = next_state
    plt.plot(state[0], state[1], 'r+')
    plt.show()
plt.savefig('final_policy_trajectory.pdf')