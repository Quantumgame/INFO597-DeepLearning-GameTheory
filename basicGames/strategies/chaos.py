import random

class Chaos():
	
	'''
	Cooperate or Defect at random.
	'''

	def strategy(self, **context):
		if random.uniform(0,1) < 0.5: return 'cooperate'
		else: return 'defect'

	def punish(self, **context):
		# nothing to learn from since it's just chaos
		pass
