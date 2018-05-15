import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
		self.env = env

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
		return self.env.action_space.sample()


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
		ob, obs, next_obs, acts, costs = [], [], [], [], [] #(horizon, num_simulated_paths, n_dim)
		[ob.append(state) for _ in range(self.num_simulated_paths)]
		for _ in range(self.horizon):
			act = []
			obs.append(ob)
			[act.append(self.env.action_space.sample()) for _ in range(self.num_simulated_paths)]
			acts.append(act)
			ob = self.dyn_model.predict(np.array(ob), np.array(act))
			next_obs.append(ob)
		costs = trajectory_cost_fn(self.cost_fn, np.array(obs), np.array(acts), np.array(next_obs))
		j = np.argmin(costs, )

		return acts[0][j]