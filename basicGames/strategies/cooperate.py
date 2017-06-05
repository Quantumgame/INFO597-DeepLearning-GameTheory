
class Cooperate:
	'''
	Always cooperate
	'''

	def strategy(self, **context):
		return 'cooperate'

	def punish(self, **context):
		# nothing to learn from since it always cooperates
		pass