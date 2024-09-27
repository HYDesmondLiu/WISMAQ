import numpy as np
import torch
import gym
import argparse
import os
import utils
import WISMAQ
import json

# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, eval_episodes=1, seed_offset=100):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	total_power = 0.
	comfort_penalty = 0.
	timesteps = 0

	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(np.array(state))
			state, reward, done, info = eval_env.step(action)
			avg_reward += reward
			total_power += info['total_power']
			comfort_penalty += info['comfort_penalty']
		timesteps += 1

	avg_reward /= eval_episodes
	with open('/data/hsinyu/01_Building/sinergym/Eplus-5Zone_buffers.json', 'r') as fp:
		Eplus5Zone_dict = json.load(fp)
		env_expert, env_random = Eplus5Zone_dict[env_name][2], Eplus5Zone_dict[env_name][1]
		normalized_score = (avg_reward - env_random)/(env_expert - env_random)

	print("---------------------------------------")
	print(f"Raw score: {avg_reward}")
	print(f"Total power: {total_power/timesteps} / Comfort penalty: {comfort_penalty/timesteps}")
	print(f"Evaluation over {eval_episodes} episodes: {normalized_score:.3f}")
	print("---------------------------------------")
	return normalized_score


if __name__ == "__main__":
	def str_to_bool(value):
		if isinstance(value, bool):
			return value
		if value.lower() in {'false', 'f', '0', 'no', 'n'}:
			return False
		elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
			return True
		raise ValueError(f'{value} is not a valid boolean value')

	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="WISMAQ")               # Policy name
	parser.add_argument("--env", default="Eplus-5Zone-hot-continuous-stochastic-v1")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=25000, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--eval_freq_on", default=2500, type=int)       # How often (time steps) we evaluate
	
	parser.add_argument("--max_timesteps_offline", default=5e4, type=int)   # Max time steps to run environment
	parser.add_argument("--max_timesteps_online", default=35_000, type=int)   # Max time steps to run environment
	parser.add_argument("--start_timesteps", default=6e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--save_model", default=1, type=int)        # Save model and optimizer parameters
	parser.add_argument("--load_model", default=1, type=int)                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=0.4, type=float)
	parser.add_argument("--normalize", default=True)
	# Off2On
	parser.add_argument("--reduced_scale", default=0.05, type=float)
	parser.add_argument("--alpha_finetune", default=0., type=float)

	parser.add_argument("--xi", default=7, type=float)
	parser.add_argument("--no_ensemble", default=10, type=int)
	parser.add_argument("--window_size", default=500, type=int)


	parser.add_argument('--lambda_energy', type=float, default=1.0)
	parser.add_argument("--buffer_folder", default='./buffers/')
	parser.add_argument("--model_path", default='./model_folder/')

	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	offline_model_file_name = f"ISMAQ_{args.env}_{args.seed}"

	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	print(f'{args}')

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	env.env.reward_fn.lambda_energy = args.lambda_energy
    
	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha,
		# WISMAQ
		"xi": args.xi,
		"no_ensemble": args.no_ensemble,
		"window_size": args.window_size,
	}

	policy = WISMAQ.WISMAQ(**kwargs)


	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_SinerGym(args.env, args.buffer_folder)

	evaluations = []
	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1

	if args.load_model:
		offline_model_file_name = 'TD3_Eplus-datacenter-hot-continuous-v1_0'
		#offline_model_file_name = f'{args.policy}_{args.env}_{args.seed}'
		policy_file = f"{args.model_path}/offline_{offline_model_file_name}"
		print(f"Load offline models: {policy_file}")
	else:
		print("Offline models training")
		# Initialize policy (offline)
		policy.ensemble_Q()
		for t in range(int(args.max_timesteps_offline)):
			policy.train(replay_buffer, args.batch_size)
			# Evaluate episode
			if (t + 1) % args.eval_freq == 0:
				print(f"Time steps: {t+1}")
				evaluations.append(eval_policy(policy, args.env, args.seed, mean, std))
				np.save(f"./results/{file_name}", evaluations)
				if args.save_model: policy.save(f"./models/offline_{file_name}")

	# Online finetuning
	eval_policy(policy, args.env, args.seed, mean, std, eval_episodes=1) 
	state, done = env.reset(), False
	reduced_size = replay_buffer.downsample(args.reduced_scale)
	replay_buffer.max_size = reduced_size
	policy.alpha = args.alpha_finetune
	policy.ensemble_Q()
	
	episode_reward = 0.
	episode_timesteps = 0
	episode_num = 0

	print(f'Online finetuning begins.')
	for t in range(int(args.max_timesteps_online)):
		episode_timesteps += 1

		if args.normalize:
			state = (state - mean) / std			

		action = (
			policy.select_action(np.array(state))
			+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
		).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps != 35040 else 0

		episode_reward += reward  

		# Store normalized state in replay buffer
		next_state_norm = (next_state - mean) / std

		# Store data in replay buffer
		if args.normalize:
			replay_buffer.add(state, action, next_state_norm, reward, done_bool)
		else:
			replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state

		policy.train(replay_buffer, args.batch_size)
		
		if done: 
			# Reset environment
			state, done = env.reset(), False

			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Episode Return: {episode_reward:.3f}")
			print(f'Qmean latest: {round(policy.Qmean_all[-1],2)} Qmean SMA ref: {round(policy.Qsma_ref,2)} Size: {replay_buffer.size} ptr: {replay_buffer.ptr}')

			episode_reward = 0.
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq_on == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed, mean, std,))
			np.save(f"./results/{file_name}", evaluations)
			np.save(f"./Qmean_{file_name}.npy", policy.Qmean_all)
			
			if args.save_model: policy.save(f"./models/online_{file_name}")

	