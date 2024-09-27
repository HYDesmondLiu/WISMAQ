import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

	def Q2(self, state, action):
		sa = torch.cat([state, action], 1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q2


class WISMAQ(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=0.4,
		xi=7,
		no_ensemble=10,
		window_size=500,
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha

		# WISMAQ
		self.xi = xi
		self.no_ensemble = no_ensemble
		self.window_size = window_size
		# Weighted QSMA
		self.psi = 0.5
		self.critic_ensemble = [{} for x in range(self.no_ensemble)]
		self.Qmean_all = np.array([])
		self.Qsma_ref = 0.

		self.total_it = 0


	def ensemble_Q(self):
		for k in range(self.no_ensemble):
			self.critic_ensemble[k]['critic'] = copy.deepcopy(self.critic)
			self.critic_ensemble[k]['critic_target'] = copy.deepcopy(self.critic_ensemble[k]['critic'])
			self.critic_ensemble[k]['critic_optimizer'] = torch.optim.Adam(self.critic_ensemble[k]['critic'].parameters(), lr=3e-4)


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def moving_average(self):
		data = self.Qmean_all[-(2*self.window_size):]
		moving_averages = []
		for i in range(len(data) - self.window_size + 1):
			window = data[i:i + self.window_size]
			window_average = sum(window) / self.window_size
			moving_averages.append(window_average)
		return moving_averages		


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		for k in range(self.no_ensemble):

			# Sample replay buffer 
			state, action, next_state, reward, not_done, idx = replay_buffer.sample_CER(batch_size)
			
			with torch.no_grad():
				# Select action according to policy and add clipped noise
				noise = (
					torch.randn_like(action) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)
				
				next_action = (
					self.actor_target(next_state) + noise
				).clamp(-self.max_action, self.max_action)

				# Compute the target Q value
				target_Q1, target_Q2 = self.critic_ensemble[k]['critic_target'](next_state, next_action)
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = reward + not_done * self.discount * target_Q

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic_ensemble[k]['critic'](state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			# Optimize the critic
			self.critic_ensemble[k]['critic_optimizer'].zero_grad()
			critic_loss.backward()
			self.critic_ensemble[k]['critic_optimizer'].step()


		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Sample replay buffer 
			state, action, next_state, reward, not_done, idx = replay_buffer.sample_CER(batch_size)

			# Compute actor loss
			pi = self.actor(state)

			# Ensemble-Q
			Q1_all = torch.FloatTensor([]).to(device)
			Q2_all = torch.FloatTensor([]).to(device)

			k = np.random.randint(self.no_ensemble)
			Q1_all = torch.cat((Q1_all, self.critic_ensemble[k]['critic'].Q1(state, pi)),1)
			Q2_all = torch.cat((Q2_all, self.critic_ensemble[k]['critic'].Q2(state, pi)),1)

			Q = torch.mean(torch.mean(torch.stack([Q2_all,Q1_all]), dim=0), dim=1)
			lmbda = 1/Q.abs().mean().detach()

			self.Qmean_all = np.append(self.Qmean_all, Q.mean().item())

			if self.total_it >= (2 * self.policy_freq * self.window_size):
				Qmean_sma = self.moving_average()[-1]
				self.Qsma_ref = self.moving_average()[-self.window_size]

				Qmean = Q.mean().clone()
				Qmean.data = torch.FloatTensor([Qmean_sma]).to(device)
				explore_term = torch.max((Qmean - self.psi * self.Qsma_ref)/(Qmean.detach() + self.Qsma_ref), torch.FloatTensor([0]).to(device))
				
				actor_loss = -lmbda * Q.mean() + self.alpha * F.mse_loss(pi, action) - self.xi * explore_term
			else:
				actor_loss = -lmbda * Q.mean() + self.alpha * F.mse_loss(pi, action) 

			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for k in range(self.no_ensemble):
				for param, target_param in zip(self.critic_ensemble[k]['critic'].parameters(), self.critic_ensemble[k]['critic_target'].parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		for k in range(self.no_ensemble):
			torch.save(self.critic_ensemble[k]['critic'].state_dict(), filename + "_critic" + f"_{k}")
			torch.save(self.critic_ensemble[k]['critic_optimizer'].state_dict(), filename + "_critic_optimizer" + f"_{k}")
			
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):

		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)