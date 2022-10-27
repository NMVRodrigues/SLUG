#from ellyn import ellyn
from sklearn.metrics import accuracy_score


class M4GP:
	def __init__(self) -> None:
		self.model = ellyn(classification=True, 
									class_m4gp=True, 
									prto_arch_on=True,
									selection='lexicase',
									fit_type='F1', # can be 'F1' or 'F1W' (weighted F1),
									g=30, # number of generations (limited by default)
									popsize=100, #population size
									verbosity=False
								)

	def fit(self,Tr_X, Tr_Y, Te_X = None, Te_Y = None):
		with open('output.log', 'a') as f:
			print('Number of features: ', Tr_X.shape[1], file=f)
		self.model.fit(Tr_X.values, Tr_Y.values)

	def predict(self, dataset):
		return self.model.predict(dataset.values)

	def getAccuracyOverTime(self):
		pass	

	def getF2OverTime(self):
		print('cenas')
		return []

	def getF2(self):
		print('cenas2')
		return 0

	def getKappaOverTime(self):
		pass

	def getAUCOverTime(self):
		pass

	def getSizeOverTime(self):
		pass

	def getBestIndividual(self):
		pass

	def getGenerationTimes(self):
		pass