import random
import tensorflow as tf
import numpy as np

def linear_annealing(n, total=8000, p_initial=1.0, p_final=0.05):
	'''
	'''
	if n >= total: return p_final
	else: return p_initial - (n * (p_initial - p_final)) / total

class QLearningNetwork:
	def __init__ (self, num_actions, obs_size = 'auto', decay=0.9, lr=0.1,
		explore_period=8000, explore_random_prob=0.2, exploit_random_prob=0.0):
		self.actions = range(num_actions)
		#self.e = 0.1
		self.id = 0
		self.e = lambda x: linear_annealing(x, explore_period,
			explore_random_prob, exploit_random_prob)
		tf.reset_default_graph()
		self.inputs1 = tf.placeholder(shape=[1,2],dtype=tf.float32, name='self.inputs1')
		self.W = tf.Variable(tf.random_uniform([2,2],0,0.01), name='self.W')
		self.Qout = tf.matmul(self.inputs1,self.W, name='self.Qout')
		self.predict = tf.argmax(self.Qout, 1, name='self.predict')

		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.nextQ = tf.placeholder(shape=[1,2],dtype=tf.float32, name='self.nextQ')
		self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
		self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
		self.updateModel = self.trainer.minimize(self.loss)

		self.allQ = None
		self.rAll = 0

		init = tf.global_variables_initializer()

		self.sess = tf.Session()
		self.sess.run(init)

	def __del__(self):
		self.sess.close()

	def choose(self, state):

		if not state:
			return random.choice(self.actions)
			#return action

		s=[state[-1]]
		'''s: state '''
		print('s', s)
		a, allQ = self.sess.run([self.predict,self.Qout], feed_dict={self.inputs1:np.identity(4)[s:s]})


		if random.random() < self.e(self.id):
			return random.choice(self.actions)

		self.id += 1
		print(a)
		return a[0]

	def learn(self, state, action, reward, new_state):
		print(state,action,reward, new_state)
		s = state[:-1]
		a = action
		r = reward
		s1 = [new_state[:-1]]

		Q1 = self.sess.run(self.Qout, feed_dict={self.inputs1:np.identity(4)[s1:s1]})
		maxQ1 = np.max(Q1)

		targetQ = self.allQ
		targetQ[0, a] = r + self.y*maxQ1

		_, W =  sess.run([updateModel, W], feed_dict={self.inputs1:np.identity(4)[s:s], nextQ:targetQ})
		self.rAll += r

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
