import pandas as pd 
import numpy as np
from scipy.stats.stats import pearsonr
import math
from sklearn.metrics import mean_squared_error

class KNNCF():
	def __init__(self, df_train_name):
		self.raw_data = pd.read_csv(df_train_name, header = None)
		self.raw_data.columns = ['movieid', 'customerid', 'rating']
		self.data = self.raw_data.pivot(index = 'customerid', columns = 'movieid', values = 'rating')
		self.sim_matrix = self.compute_similarity_matrix()
		
		
		self.user_means = self.data.mean(axis = 1)
		self.central_mean = self.user_means.mean()
		self.norm_data = self.data.sub(self.user_means, axis=0)
		self.users = self.user_means.index
		self.movies = self.norm_data.columns


	# user based
	def compute_similarity_matrix(self):
		return self.data.T.corr().fillna(0)

	def choose_candidates(self, user, item):
		norm_data_js = self.norm_data.loc[:, item]
		user_js_no = norm_data_js[norm_data_js.isnull()].index
		w = self.sim_matrix.loc[user,:].copy()
		w.loc[user_js_no] = None
		return w.dropna()

	def select_k_neighbors(self, k, user, item):
		# get the sim score between user and each candidate
		can_score = self.choose_candidates(user, item)
		# sort the series
		can_score = can_score.sort_values(ascending = False)
		can_score.drop(index = user, inplace = True)
		return can_score[:k]


	def predict(self, k, user_i, item):
		selection_res = self.select_k_neighbors(k, user_i, item)
		r_hat = 0
		sim_sum = 0
		for user, sim in selection_res.iteritems():
			rating = self.data.loc[user, item]
			r_hat += rating * sim
			sim_sum += sim

		return r_hat / sim_sum

knncf = KNNCF("train.txt")
test = pd.read_csv("train.txt", header = None)
test.columns = ['movieid', 'customerid', 'rating']
test_data = test.pivot(index = 'customerid', columns = 'movieid', values = 'rating')
preds = []
ys = []
for i in range(0, 10):
	movie_k, user_i, = [int(test.iat[i,0]), int(test.iat[i,1])]
	prediction = knncf.predict(3, user_i, movie_k)
	preds.append(prediction)
	ys.append(test_data.loc[user_i, movie_k])

print(mean_squared_error(ys, preds))
