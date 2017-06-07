
class Grim:
	'''
    Cooperate unless the opponent defects, in which case, defect forever.
    '''
	def __init__(self, reward=4, temptation=5, penalty=1, sucker=0):
		self.grim = False
		self.actions = []

		

	def punish(self, **context):
		#No learning is done
		pass

	def strategy(self, state, sucker=1, temptation=5, **context):

		if not state: # if there's no history
			return 'cooperate'
		elif self.grim:
			return 'defect'
		elif not self.grim and state[-1] in (1,5):
			self.grim = True
			return 'defect'
		else:
			return 'cooperate'

