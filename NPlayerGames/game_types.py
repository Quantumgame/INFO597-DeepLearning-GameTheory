import random
from tqdm import tqdm

from strategies import chaos as c
from strategies import cooperate as cp
from strategies import defect as d
from strategies import machine_learning as ml
from strategies import pavlov as p
from strategies import grim as g
from strategies import tit_for_tat as t


class NPlayerGame:
	def check_params(self, n_players):
		'''
		Make sure that Parameters are correct otherwise I need to throw an error.
		n_players : Int containing number of players
		n_actions : Int containing number of actions
		'''
		if n_players < 2 or n_players is None: # Raise error
			pass

	def __init__(self, n_players=2, agent_strategies=None):
		'''
		agent_strategies : dict of agent names with value of strings containing valid strategies
		util_vals : dict of actions with int values for number
		'''
		self.n_players = n_players  # number of players
		n_actions=4  # hard set for now, hard to implement dynamic action game
		self.check_params(n_players)  # Make sure params are valid

		# Create agents
		#TODO assume passed in dict is correct for now
		if agent_strategies is None: self.agents = self.create_agents(n_players)  # create agents
		else: self.agents = agent_strategies

		self.data = self.create_data()
		self.id = 0
		self.actions = {} # store actions 'peace, war, etc.'
		self.utils = {} # store actual values

	def create_data(self):
		data = {'id': []}
		for agent in self.agents: data[agent] = []
		return data

	def create_agents(self, n_players):
		'''
		create agents of random type
		n_players : integer for number of players
		returns dictionary of agent objects
		'''
		agents = {}
		strategy_types = ['chaos', 'defect', 'qlearn', 'deepqlearn', 'grim', 'pavlov', 'titfortat', 'cooperate']
		#strategy_types = ['chaos', 'defect', 'deepqlearn']#'qlearn', 'grim', 'pavlov', 'titfortat', 'cooperate']
		#strategy_types = ['cooperate']
		dq = False
		for n in range(1, n_players+1):
			_type = random.choice(strategy_types)
			if _type == 'chaos': agents['agent{0}_chaos'.format(str(n))] = c.Chaos()
			elif _type == 'defect': agents['agent{0}_defect'.format(str(n))] = d.Defect()
			elif _type == 'qlearn': agents['agent{0}_qlearn'.format(str(n))] = ml.QLearn()
			elif _type == 'deepqlearn' and dq is not True:
				dl = ml.DeepQLearner(3, 1, decay=0.9, learning_rate=0.02)
				agents['agent{0}_deepqlearn'.format(str(n))] = ml.QLearn(agent=dl)
				dq=True
			elif _type == 'grim': agents['agent{0}_grim'.format(str(n))] = g.Grim()
			elif _type == 'pavlov': agents['agent{0}_pavlov'.format(str(n))] = p.Pavlov()
			elif _type == 'titfortat': agents['agent{0}_titfortat'.format(str(n))] = t.Titfortat()
			elif _type == 'cooperate': agents['agent{0}_cooperate'.format(str(n))] = cp.Cooperate()
		else: print(_type)#print ('Error for {}').format(_type) #TODO Throw error
		return agents

	def get_results(self, save=True):
		'''
		save : saves util values to self.data
		'''
		#print('self.actions', self.actions)
		for current_player_name in self.actions:
			current_player_action = self.actions[current_player_name]
			for opponent_name in self.actions:
				opponent_action = self.actions[opponent_name]
				if opponent_name == current_player_name:
					pass  # We don't want compare values with ourselves
				else:
					# First 3 cases current player chose to cooperate
					if opponent_action == 'defect' and  current_player_action == 'cooperate':
						self.utils[current_player_name] = 1
						break # you can't receive a higher utility score because someone is wagining war break of of checking oppoonents
					elif opponent_action == 'compromise' and current_player_action == 'cooperate':
						self.utils[current_player_name] = 2
					elif opponent_action == 'cooperate' and current_player_action == 'cooperate':
						if self.actions[current_player_name] == 2: pass # if someone compromised then they cannot get a utility value of 4
						else: self.utils[current_player_name] = 4  # So far no one has waged war or proposed a compromise, only peaceful solutions

					# Second 2 cases current player chose to compromise
					elif current_player_action == 'compromise' and (opponent_action == 'cooperate' or opponent_action == 'compromise'):
						self.utils[current_player_name] = 3
					elif opponent_action == 'defect' and current_player_action == 'compromise':
						self.utils[current_player_name] = 2
						break # you can't receive a higher utility score because someone is wagining war break off of  checking oppoonents

					# Second 2 cases current player chose to defect
					elif current_player_action == 'defect' and (opponent_action == 'cooperate' or opponent_action == 'compromise'):
						self.utils[current_player_name] = 5
					elif opponent_action == 'defect' and current_player_action == 'defect':
						self.utils[current_player_name] = 2
						break # you can't receive a higher utility score because someone is wagining war break off of  checking oppoonents
					else:
						print('{2} did not handle case for this action: {0} and opponent action {1}'.format(str(current_player_action), str(opponent_action), str(current_player_name)))
			#check_opponents(current_player_name, current_player_action)

		if save == True:
			for agent in self.utils:
				self.data[agent].append(self.utils[agent])
			self.data['id'].append(self.id)
			self.id += 1


	def play(self, moves=1):
		for _ in tqdm(range(0, moves), total=moves, desc='Playing Game'):
			for agent in self.agents:  # Play all games and get actions from all users
				self.actions[agent] = self.agents[agent].strategy(state=self.data[agent])


			self.get_results(save=True) # Save the results of the game

			#print('self.agents', self.agents)
			#print('self.utils', self.utils)
			for agent in self.agents: # each agent should learn from others after they all have played or can this be indented?
				self.agents[agent].punish(state=self.data[agent][:-1],
				 action=self.actions[agent],
				 reward=self.utils[agent],
				 new_state=self.data[agent])
