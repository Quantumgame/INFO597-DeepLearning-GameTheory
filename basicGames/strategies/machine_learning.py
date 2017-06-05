from .q_learning import QLearning as ql

class QLearn:
	def __init__(self):
		self.agent = ql(2) # Cooperate or defect only 2 states
		self.last_action = None # No action when initalized
		self.actions = []

	def strategy(self, state):
		set_of_actions = {0:'defect', 1: 'cooperate'}
		action = set_of_actions[self.agent.choose(state=state)]
		self.actions.append(action)
		return action

	def punish(self, state, action, reward, new_state):
		if self.last_action:
			self.agent.learn(state=state, action=action, reward=reward, new_state=new_state)
		self.last_action = True


