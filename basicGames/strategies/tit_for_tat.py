
class Titfortat():
	'''Cooperate at first, and play the opponents previous move after.'''
	def __init__(self, reward=4, temptation=5, penalty=1, sucker=0):
		self.id = 0
		self.actions = []

		self.REWARD = reward
		self.TEMPTATION = temptation
		self.PENALTY = penalty
		self.SUCKER = sucker

	def strategy(self, **context):
		self.id += 1
		if self.id <= 1:
			self.actions.append("cooperate")
			return "cooperate"

		if state[-1] in (self.TEMPTATION, self.REWARD):
			self.actions.append("cooperate")
			return "cooperate"
		else: # if history[-1] == (1 or 5): # or -1
			self.actions.append("defect") 
			return "defect"

	def punish(self, **context):
		# nothing is done because Titfortat strategy does not learn
		pass
