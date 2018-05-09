#!/usr/bin/env python

"""
Behavorial cloning of an expert policy using a simple Feedforward Neural Network.
Example usage:
    python behavioral_cloning.py experts/Humanoid-v2.pkl Humanoid-v2 data_experts/Humanoid-v2_20_data.pkl --num_rollouts 20 #--render
"""

import pickle
import numpy as np
import tensorflow as tf
import tf_util
import gym
import load_policy
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"    

from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
# from keras.utils import np_utils


# load expert collected data
def load_data(filename):
	with open(filename, 'rb') as f:
		data = pickle.loads(f.read())
	return data

def main():
	# add arguments/parameters to run command
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('data_file', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
	parser.add_argument("--max_timesteps", type=int)
	args = parser.parse_args()

	print('loading and building expert policy')
	policy_fn = load_policy.load_policy(args.expert_policy_file)
	print('loaded and built')

	# setting task
	task = args.envname
	task_data = args.data_file

	# load in expert policy observation data
	data = load_data(task_data)
	obs_data = np.array(data['observations'])
	act_data = np.array(data['actions'])
	
	# split data into train and test set
	n = obs_data.shape[0]
	#Shuffle arrays or sparse matrices in a consistent way
	obs_data, act_data = shuffle(obs_data, act_data, random_state=0)
	split_value = int(n * 0.8) # 80/20 split
	X_train = np.array(obs_data[:split_value])
	X_test = np.array(obs_data[split_value:])
	y_train = np.array(act_data[:split_value])
	y_test = np.array(act_data[split_value:])

	# reshape data 
	X_train = X_train.reshape(X_train.shape[0], obs_data.shape[1])
	X_test = X_test.reshape(X_test.shape[0], obs_data.shape[1])
	Y_train = y_train.reshape(y_train.shape[0], act_data.shape[2])
	Y_test = y_test.reshape(y_test.shape[0], act_data.shape[2])

	# create a feedforward neural network
	model = Sequential() # the Sequential model is a linear stack of layers
	model.add(Dense(128, activation='relu', input_shape=(obs_data.shape[1],))) # regular densely-connected NN layer
	model.add(Dense(128, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(act_data.shape[2], activation='linear'))
	model.compile(loss='msle', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, Y_train, batch_size=64, nb_epoch=20, verbose=1)
	score = model.evaluate(X_test, Y_test, verbose=1)
	model.save('models/' + task + '_cloned_model.h5')
	
	with tf.Session():
		tf_util.initialize()
		env = gym.make(args.envname)
		max_steps = args.max_timesteps or env.spec.timestep_limit

		returns = []
		new_observations = []
		new_exp_actions = []

		model = load_model('models/' + task + '_cloned_model.h5')

		for i in range(args.num_rollouts):
			print('iter', i)
			obs = env.reset()
			done = False
			totalr = 0.
			steps = 0
			while not done:
				obs = np.array(obs)
				exp_action = policy_fn(obs[None,:])
				# obs = obs.reshape(1, len(obs), 1)
				obs = obs.reshape(1, len(obs))
				action = (model.predict(obs, batch_size=64, verbose=0))
				# print 'obs: ' + str(obs)
				# print "predicted: " + str(action)
				# print "expert: " + str(exp_action)

				new_observations.append(obs)
				new_exp_actions.append(exp_action)
				obs, r, done, _ = env.step(action)
				totalr += r
				steps += 1
				if args.render:
					env.render()
				if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
				if steps >= max_steps:
					break
			returns.append(totalr)

		print('returns', returns)
		print('mean return', np.mean(returns))
		print('std of return', np.std(returns))


if __name__ == '__main__':
	main()