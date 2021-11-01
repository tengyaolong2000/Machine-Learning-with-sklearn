import os
import pickle

class DTree():
	
	def __init__(self):
		path = '/Users/tengyaolong/Desktop/Anaconda stuff/Learning/final_model.pkl'
		file = open(path, 'rb')
		self.model = pickle.load(file)


	def predict(self, age, job, marital, education, default, balance, 
				housing, loan, day, month, duration, campaign, pdays, 
				previous):
		X = [age, job, marital, education, default, balance, 
				housing, loan, day, month, duration, campaign, pdays, 
				previous]
		return self.model.predict([X])