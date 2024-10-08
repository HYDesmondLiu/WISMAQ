import numpy as np
import torch
import copy

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_prob = 1.2, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			ind,
		)


	def sample_CER(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		ind[-1] = copy.deepcopy(self.ptr-1)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			ind,
		)


	def downsample(self, reduced_scale):
		reduced_size = int(self.size * reduced_scale)
		ind = np.random.randint(0, self.size, size=reduced_size)
		print(f'Replay buffer size reduced from:{self.size} to {reduced_size}')
		self.state[:reduced_size] = self.state[ind]
		self.action[:reduced_size] = self.action[ind]
		self.next_state[:reduced_size] = self.next_state[ind]
		self.reward[:reduced_size] = self.reward[ind]
		self.not_done[:reduced_size] = self.not_done[ind]

		self.state[reduced_size:] = 0
		self.action[reduced_size:] = 0
		self.next_state[reduced_size:] = 0
		self.reward[reduced_size:] = 0
		self.not_done[reduced_size:] = 0

		self.size = reduced_size
		self.ptr = (reduced_size) % self.max_size
		return reduced_size


	def convert_SinerGym(self, env_name, buffer_folder):
		state = np.load(f'{buffer_folder}/{env_name}_state.npy')
		action = np.load(f'{buffer_folder}/{env_name}_action.npy')
		next_state = np.load(f'{buffer_folder}/{env_name}_next_state.npy')
		reward = np.load(f'{buffer_folder}/{env_name}_reward.npy')
		not_done = np.load(f'{buffer_folder}/{env_name}_not_done.npy')

		self.state = state
		self.action = action
		self.next_state = next_state
		self.reward = reward
		self.not_done = not_done
		self.size = self.state.shape[0]


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std

