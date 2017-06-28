from game_types import NPlayerGame

from strategies import tit_for_tat as tt
from strategies import chaos as c
from strategies import defect as d
from strategies import machine_learning as ml
from strategies import pavlov as p
from strategies import grim as g

#import seaborn as sns
import matplotlib.pyplot as plt
#sns.set()


def main():
	game = NPlayerGame(n_players=10)
	game.play(300)
	#a = plt.plot(game.data['A'], 'r-.', label='a')
	#b = plt.plot(game.data['B'], 'b-.', label='b')
	#plt.legend([a, b], ['a', 'b'])
	#plt.show()
	#print(game.data)



if __name__ == '__main__':
	main()
