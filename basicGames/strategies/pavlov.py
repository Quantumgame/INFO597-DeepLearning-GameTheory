
class Pavlov:
    '''
    Cooperate until punished then switch strategies.
    '''
	def __init__(self):
		self.defaultstrategy = 'cooperate'

	def punish(self, **context):
		#No learning is done
		pass

	def strategy(self, history):
		if not history:
			return self.defaultstrategy
		elif history[-1] in (1,5) and self.defaultstrategy == 'cooperate':
			self.defaultstrategy = 'defect'
		elif history[-1] in (1,5) and self.defaultstrategy == 'defect':
			self.defaultstrategy = 'cooperate'
		return self.defaultstrategy
