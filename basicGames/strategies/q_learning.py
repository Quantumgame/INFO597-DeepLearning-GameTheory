import random

def linear_annealing(n, total=8000, p_initial=1.0, p_final=0.05):
	'''
	'''
	if n >= total: return p_final
	else: return p_initial - (n * (p_initial - p_final)) / total

class QLearning:
	def __init__ (self, num_actions, obs_size = 'auto', decay=0.9, lr=0.1,
		explore_period=8000, explore_random_prob=0.2, exploit_random_prob=0.0):
		
		self.id = 0
		self.values = {}
		self.lr = lr
		self.decay = decay
		self.explore_period = explore_period
		self.exploit_random_prob = exploit_random_prob
		self.explore_random_prob = explore_random_prob

		self.epsilon = lambda x: linear_annealing(x, self.explore_period,
			self.explore_random_prob, self.exploit_random_prob)
		self.actions = range(num_actions)
		self.obs_size = obs_size

	def learn(self, state, action, reward, new_state):
		#print(state)
		#print(action)
		if (new_state[-1], action) not in self.values:
			self.values[(new_state[-1], action)] = reward
		
		if self.obs_size == 'auto':
			self.obs_size = len(state)
		state = tuple(state[-self.obs_size:])
		new_state = tuple(new_state[-self.obs_size:])
		old_value = self.values.get((new_state[-1], action))

		actions = []
		for _action in self.actions:
			actions.append( self.values.get((new_state, _action), 0.5) )
		qmax = max(actions)

		#print(self.values)
		#print(qmax)
		#print(old_value)
		#new_value = 1 + self.lr * (reward + self.decay * qmax - 1) # Works

		new_value = old_value + self.lr * (reward + self.decay * qmax - old_value)
		
		self.values[state, action] = new_value

	def choose(self, state):
		#TODO update self.values
 
		if not state:
			action = random.choice(self.actions)
			return action

		if self.obs_size == 'auto':
			self.obs_size = len(state)
		state = tuple(state[-self.obs_size:])
		
		self.id += 1
		if random.random() < self.epsilon(self.id):
			return random.choice(self.actions)

		return max(self.actions, key=lambda action: self.values.get((state, action), 0.5))
