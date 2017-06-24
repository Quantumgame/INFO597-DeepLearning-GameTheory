from .q_learning import QLearning as ql
from .q_learning import QLearningNetwork as qln
import tensorflow as tf

class QLearn:
	def __init__(self, obs_size='auto', decay=0.9, lr=0.1,
		explore_period=8000, explore_random_prob=0.2, exploit_random_prob=0.0, agent_type='table'):

		if agent_type is 'table':
			self.agent = ql(2, obs_size, decay, lr,
			explore_period, explore_random_prob, exploit_random_prob) # Cooperate or defect only 2 states
		elif agent_type is 'net':
			self.agent = qln(2)
		elif agent_type is 'deepnet':
			# self.agent = dql(2)
			pass
			self.agent = qln(2)
			#
		else:
			self.agent = ql(2)
		self.last_action = None # No action when initalized
		self.actions = []

	def strategy(self, state, **context):
		set_of_actions = {0:'defect', 1: 'cooperate'}
		action = set_of_actions[self.agent.choose(state=state)]
		self.actions.append(action)
		return action

	def punish(self, state, action, reward, new_state):
		if self.last_action:
			self.agent.learn(state=state, action=action, reward=reward, new_state=new_state)
		self.last_action = True
