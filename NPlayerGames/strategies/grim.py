"""
import random

class Grim:
	'''
    Cooperate until opponent defects  defect forever. Compromise  if opponent compromises once (not cooperate). Defect if any agent defects or if opponenets Compromise more than self.patience
    '''
	def __init__(self, reward=4, temptation=5, penalty=1, sucker=0):
		self.grim = False
		self.actions = []
		self.patience = None
		self.count = 0
		self.strategy = 'cooperate'


	def punish(self, **context):
		#No learning is done
		pass

	def strategy(self, state, sucker=1, temptation=5, patience=None, **context):

		def how_grim(patience):
			if patience is None: self.patience = int(random.random() / random.random())
			else: self.patience = patience
			if self.patience == 0 : self.patience = 1

		if self.patience is None: how_grim(patience)

		if not state: # if there's no history
			return self.strategy
		elif self.grim: # if I was screwed always defect no need to check other statements
			return 'defect'

		elif not self.grim and state[-1] in (1,2):  # if I was screwed in the last round
			if state[-1] == 1:  # always defect
				self.grim = True
				self.strategy = 'defect'
				return 'defect'
			else: # If no one defected they compromised
				self.count +=1  # increase count
				if self.count > self.patience:  # if count is higher than patience
					self.grim = True  # always defect now
					return 'defect'
				self.strategy = 'compromise'  # defaultstrategy is compromise now
				return self.strategy
		else:  # no one has defected yet...
			return self.strategy
"""

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
