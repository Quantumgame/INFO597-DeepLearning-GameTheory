
class PrisonersDilemma:

	def __init__(self, prisoner_a, prisoner_b, reward=4, 
		temptation=5, penalty=1, sucker=0):
		'''
		'''
		# Utility Values
		self.REWARD = reward
		self.TEMPTATION = temptation
		self.PENALTY = penalty
		self.SUCKER = sucker

		# Players
		self.prisoner_a = prisoner_a
		self.prisoner_b = prisoner_b
		self.data = {'id': [], 'A': [], 'B': []}
		self.id = 0

	def play(self, moves=1):
		for _ in range(0, moves):
			prisoner_a_action = self.prisoner_a.strategy(state=self.data['A'])
			prisoner_b_action = self.prisoner_b.strategy(state=self.data['B'])

			if prisoner_a_action == 'defect' and prisoner_b_action == 'defect':
				reward_a = self.PENALTY
				reward_b = self.PENALTY
			elif prisoner_a_action == 'defect' and prisoner_b_action == 'cooperate':
				reward_a = self.TEMPTATION
				reward_b = self.SUCKER
			elif prisoner_b_action == 'defect' and prisoner_a_action == 'cooperate':
				reward_b = self.TEMPTATION
				reward_a = self.SUCKER
			elif prisoner_a_action == 'cooperate' and prisoner_b_action == 'cooperate':
				reward_a = self.REWARD
				reward_b = self.REWARD
			else: # illegal move
				pass
			
			self.id +=1
			self.data['id'].append(self.id)
			self.data['A'].append(reward_a)
			self.data['B'].append(reward_b)


			self.prisoner_a.punish(state=self.data['A'][:-1], action=prisoner_a_action, reward=reward_a, new_state=self.data['A'])
			self.prisoner_b.punish(state=self.data['B'][:-1], action=prisoner_b_action, reward=reward_b, new_state=self.data['B'])

