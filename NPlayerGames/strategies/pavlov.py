
class Pavlov:
	'''
    Cooperate until punished then switch strategies.
    '''
	def __init__(self):
		self.defaultstrategy = 'cooperate'
		self.defaultstrategies = {1: 'cooperate', 2: 'defect', 3: 'compromise'}

	def punish(self, **context):
		#No learning is done
		pass

	def strategy(self, state, temptation=5, sucker=1):

		if not state:
			return self.defaultstrategy
		elif state[-1] == 1 and self.defaultstrategy == 'cooperate':
			self.defaultstrategy = 'defect'
		elif state[-1] == 1 and self.defaultstrategy == 'defect':
			self.defaultstrategy = 'compromise'
		elif state[-1] == 1 and self.defaultstrategy == 'compromise':
			self.defaultstrategy = 'cooperate'
		return self.defaultstrategy
