from PNCF import PNCF
import pandas as pd 
import numpy as np
from scipy.stats.stats import pearsonr
import math
from sklearn.metrics import mean_squared_error
from math import sqrt


pncf = PNCF("train.txt")
test = pd.read_csv("train.txt", header = None)
test.columns = ['movieid', 'customerid', 'rating']
test_data = test.pivot(index = 'customerid', columns = 'movieid', values = 'rating')

ks = [5, 7, 9]
# perturb on similarity
for k in ks:
	preds = []
	ys = []
	for i in range(0, 20):
		movie_k, user_i, = [int(test.iat[i,0]), int(test.iat[i,1])]
		prediction = pncf.predict(k, user_i, movie_k, w = 0.01, eps = 3, B = 1, state = 0)
		preds.append(prediction)
		ys.append(test_data.loc[user_i, movie_k])
	print("Perturbation on similarity")
	print ("k %5d" %k)
	print("MAE: %.5f" %sqrt(mean_squared_error(ys, preds)))


# perturb on ratings
for k in ks:
	preds = []
	ys = []
	for i in range(0, 20):
		movie_k, user_i, = [int(test.iat[i,0]), int(test.iat[i,1])]
		prediction = pncf.predict(k, user_i, movie_k, w = 0.01, eps = 3, B = 1, state = 1)
		preds.append(prediction)
		ys.append(test_data.loc[user_i, movie_k])
	print("Perturbation on rating")
	print ("k %5d" %k)
	print("MAE: %.5f" %sqrt(mean_squared_error(ys, preds)))


knncf = KNNCF("train.txt")

for k in ks:
	preds = []
	ys = []
	for i in range(0, 20):
		movie_k, user_i, = [int(test.iat[i,0]), int(test.iat[i,1])]
		prediction = knncf.predict(k, user_i, movie_k)
		preds.append(prediction)
		ys.append(test_data.loc[user_i, movie_k])
	print("Naive knn")
	print ("k %5d" %k)
	print("MAE: %.5f" %sqrt(mean_squared_error(ys, preds)))




