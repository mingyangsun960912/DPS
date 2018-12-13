import pandas as pd 
import numpy as np
from scipy.stats.stats import pearsonr
import math
from sklearn.metrics import mean_squared_error
from math import sqrt

class PNCF():
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

	def select_k_private_neighbors(self, k, user, item, w, eps, B):
		# get the sim score between user and each candidate
		can_score = self.choose_candidates(user, item)
		# sort the series
		can_score = can_score.sort_values(ascending = False)
		# get the kth score
		pos = 0
		s_k = can_score.values[k]

		low_score_bound = s_k - w
		# c1, all candidate >= s_k - w
		c1 = can_score[lambda val : val >= low_score_bound]
		c1 = c1.drop(index = user)
		c1_orig = c1.copy()
		# c0, all candidates < s_k - w
		c0 = can_score[lambda val : val < low_score_bound]
		# change score in c0 to be s_k - w
		c0[:] = low_score_bound
		c0_orig = c0.copy()
		ks = []
		rs_dict = {}
		for i in range(0, k):

			# calculate psum
			p_sum = 0
			probs = []
		
			for user_j, score in c1.iteritems():
				rs = self.calculate_rs(user, user_j, B)
				rs_dict[user_j] = rs
				prob = self.calculate_prob(eps, score, k, rs)
				probs.append(prob)
				p_sum += prob
			for user_j, score in c0.iteritems():

				rs = self.calculate_rs(user, user_j, B)
				rs_dict[user_j] = rs
				prob = self.calculate_prob(eps, score, k, rs)
				probs.append(prob)
				p_sum += prob

			# select item based on prob
			combo = c1.copy().append(c0.copy())
			probs = [prob / p_sum for prob in probs]
			selected_user = np.random.choice(combo.index, p = probs)
			ks.append(selected_user)
			# delete
			if selected_user in c1.keys():
				c1 = c1.drop(index = selected_user)
			else:
				c0 = c0.drop(index = selected_user)

		return (ks, c1_orig, c0_orig, rs_dict)

	def calculate_prob(self, eps, score, k, rs):
		return math.exp(eps * score /(4 * k * rs))

	def calculate_rs(self, user_i, user_j, B):
		# delete a movie, see how the score changes

		norm_data_irs = self.norm_data.loc[user_i, :]
		norm_data_jrs = self.norm_data.loc[user_j, :]


		ui_nr_idx = norm_data_irs[norm_data_irs.notnull()].index
		uj_nr_idx = norm_data_jrs[norm_data_jrs.notnull()].index
		inter_items = ui_nr_idx.intersection(uj_nr_idx)

		max_c = 0
		for selected_item in inter_items:
		
			mod_ndi = norm_data_irs.drop(index = selected_item, inplace = False)
			mod_ndj = norm_data_jrs.drop(index = selected_item, inplace = False)
			mod_ndj = mod_ndj.fillna(0)
			mod_ndi = mod_ndi.fillna(0)
			orig_score = pearsonr(norm_data_jrs.fillna(0).values, norm_data_irs.fillna(0).values)[0]
			changed_score = pearsonr(mod_ndi.values, mod_ndj.values)[0]

			dif = abs(orig_score - changed_score)
			if dif > max_c:
				max_c = dif
		# return smoothed recommendation awareness sensitivity
		#return math.exp(-B) * dif
		return math.exp(-B) * max_c

	def predict_sim_perturbation(self, eps, kneighbors, c1, c0, item, rs_dict):
		# k, user, item, w, eps, B
		# add noise on similarity
		customers = kneighbors
		sim_scores = []
		for cid in customers:
			if cid in c1.keys():
				sim_scores.append(c1[cid])
			else:
				sim_scores.append(c0[cid])

		ratings = []
		for cid in customers:

			rating = self.data.loc[cid, item]
			ratings.append(rating)

		# add laplace noise to similarity
		perturbed_sim_scores = []
		for i in range(0, len(sim_scores)):
			sim_score = sim_scores[i]
			sim_score += sim_score * np.random.laplace(scale = 2 * rs_dict[customers[i]] / eps)
			perturbed_sim_scores.append(sim_score)

		sim_sum = 0
		rating = 0
		print(ratings)
		print(sim_scores)
		print(perturbed_sim_scores)
		for i in range(0, len(perturbed_sim_scores)):
			pss = perturbed_sim_scores[i]
			sim_sum += pss
			rating += pss * ratings[i]

		if sim_sum == 0:
			sim_sum = 1
		return rating / sim_sum


	def predict_rating_perturbation(self, eps, kneighbors, c1, c0, item, rs_dict):
		customers = kneighbors
		sim_scores = []
		for cid in customers:
			if cid in c1.keys():
				sim_scores.append(c1[cid])
			else:
				sim_scores.append(c0[cid])

		ratings = []
		for cid in customers:
			rating = self.data.loc[cid, item]
			ratings.append(rating)

		# add laplace noise to similarity

		perturbed_rating_scores = []
		for i in range(0, len(ratings)):
			r_score = ratings[i]
			r_score += r_score * np.random.laplace(scale = 2 * rs_dict[customers[i]] / eps)
			perturbed_rating_scores.append(r_score)

		sim_sum = 0
		rating = 0
		for i in range(0, len(perturbed_rating_scores)):
			prs = perturbed_rating_scores[i]

			sim_sum += sim_scores[i]
			rating += prs * sim_scores[i]
		if sim_sum == 0:
			sim_sum = 1
		return rating / sim_sum

	def predict(self, k, user_i, item, w, eps, B, state = 0):
		selection_res = self.select_k_private_neighbors(k, user_i, item, w, eps, B)
		neighbors = selection_res[0]
		c1 = selection_res[1]
		c0 = selection_res[2]
		rs_dict = selection_res[3]
		r_hat = 0
		if state == 0:
			# perturb on similarity
			r_hat = self.predict_sim_perturbation(eps, neighbors, c1, c0, item, rs_dict)

		else:
			# perturb on ratings
			r_hat = self.predict_rating_perturbation(eps, neighbors, c1, c0, item, rs_dict)
		print(r_hat)
		return r_hat



