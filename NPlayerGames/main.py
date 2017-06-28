from game_types import PrisonersDilemma
from strategies import tit_for_tat as tt
from strategies import chaos as c
from strategies import machine_learning as ml

#import seaborn as sns
import matplotlib.pyplot as plt
#sns.set()

def tt_vs_tt():
	'''
	tit for tat against tit for tat
	'''
	player1 =  tt.Titfortat()
	player2 = tt.Titfortat()
	game = PrisonersDilemma(player1, player2)
	game.play(100)

	print(game.data)

def tt_vs_chaos():
	player1 =  tt.Titfortat()
	player2 = c.Chaos()
	game = PrisonersDilemma(player1, player2)
	game.play(100)

	print(game.data)
	pass

def ml_vs_chaos():
	player1 = ml.QLearn()
	player2 = c.Chaos()
	game = PrisonersDilemma(player1, player2)
	game.play(10000)
	a = plt.plot(game.data['A'], 'r-.', label='a')
	b = plt.plot(game.data['B'], 'b-.', label='b')
	plt.legend([a, b], ['a', 'b'])
	plt.show()
	print(game.data)

def main():
	#tt_vs_tt()
	#tt_vs_chaos()
	ml_vs_chaos()

if __name__ == '__main__':
	main()