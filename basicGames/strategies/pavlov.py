
class Pavlov:
	'''
    Cooperate until punished then switch strategies.
    '''
	def __init__(self):
		self.defaultstrategy = 'cooperate'

	def punish(self, **context):
		#No learning is done
		pass

	def strategy(self, state, temptation=5, sucker=1):

		if not state:
			return self.defaultstrategy
		elif state[-1] == sucker and self.defaultstrategy == 'cooperate':
			self.defaultstrategy = 'defect'
		elif state[-1] == sucker and self.defaultstrategy == 'defect':
			self.defaultstrategy = 'cooperate'
		return self.defaultstrategy
