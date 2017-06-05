
class Defect:
	'''
	Always defect
	'''

	def strategy(self, **context):
		return 'defect'

	def punish(self, **context):
		# nothing to learn from since it always defects
		pass
		