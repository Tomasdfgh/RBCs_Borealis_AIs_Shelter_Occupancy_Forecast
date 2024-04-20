from sklearn.datasets import load_breast_cancer
import numpy as np
from matplotlib import pyplot as plt

def k_means(data,k):

	#---Initialization---#
	# Cluster centroids placement are picked at a data point
	cens_cord = []
	ran_lst = []
	while len(set(ran_lst)) < k:
		num = np.random.randint(0,len(data))
		ran_lst.append(num)
	ran_lst = set(ran_lst)
	
	for i in range(0,k):
		cens_cord.append(data[ran_lst.pop()])

	#---End of Initialization---#

	#Repeating loop
	stor_cord = []
	temporary = []
	for i in range(len(cens_cord)):
		for z in range(len(cens_cord[0])):
			temporary.append(cens_cord[i][z])
		stor_cord.append(temporary)
		temporary = []

	while True:
		cen_closest = []
		dist = []
		dist_2 = []
		temp = []
		sum_0 = 0

		#---Cluster Assignment Step---#

		for d in range(0,k):
			for i in range(len(data)): #569
				for z in range(len(data[i])): #30
					sum_0 += (data[i][z] - cens_cord[d][z]) ** 2
				dist.append(np.sqrt(sum_0))
				sum_0 = 0

		for i in range(0,k):
			for z in range(len(data)):
				temp.append(dist.pop(0))
			dist_2.append(temp)
			temp = []


		for i in range(len(data)):
			cen_closest.append(find_least(dist_2,i))

		#---End of Cluster Assignment---#

		#---Move Centroid---#

		mean = []
		for i in range(len(data[0])):
			mean.append(0)

		mean_counter = 0
		for i in range(0,k):
			q = 0
			while q < len(data):
				if cen_closest[q] == i:
					mean_counter += 1
					for p in range(len(data[0])):
						mean[p] += data[q][p]
				q += 1
			if mean_counter != 0:
				for o in range(len(data[0])):
					mean[o] = float(mean[o])/mean_counter
				cens_cord[i] = mean
				mean = []
				for l in range(len(data[0])):
					mean.append(0)
				mean_counter = 0

		#--- End of Move Centroid---#

		#---Terminating Conditions---#
		if cens_cord == stor_cord:
			return cens_cord,cen_closest
		else:
			temporary = []
			stor_cord = []
			for i in range(len(cens_cord)):
				for z in range(len(cens_cord[0])):
					temporary.append(cens_cord[i][z])
				stor_cord.append(temporary)
				temporary = []

		#---End of Terminating Conditions---#

def find_max(data, i):
	#i is the index of row
	#this function is to find the max for the specified row
	max_1 = 0
	for z in range(len(data)):
		if data[z][i] > max_1:
			max_1 = data[z][i]
	return max_1

def find_least(dist,i):
	lst = []
	for z in range(len(dist)):
		lst.append(dist[z][i])
	return lst.index(min(lst))

if __name__ == '__main__':
	data = load_breast_cancer()
	data_2c = [[1,3],[2,4],[3,2],[4,1],[5,2],[3,5],[30,15],[31,14],[33,17],[31,12],[35,16],[32,15]]
	data_3c = [[1,3],[2,4],[3,2],[4,1],[5,2],[3,5],[30,15],[31,14],[33,17],[31,12],[35,16],[32,15],[9,30],[12,27],[8,29],[10,26],[6,31],[9,25]]
	data_3c_3d = [[1, 1, 1], [1, 2, 1], [2, 1, 2], [-1, 2, -3], [-2, 2, 1], [0, 0, 0],[34, 56, 1], [29, 54, 3], [33, 58, 3], [30, 50, -1], [33, 59, -4], [31, 52, 2],[-62, -53, 30], [-59, -56, 31], [-60, -60, 34], [-55, -59, 29], [-57, -66, 25], [-50, -61, 32]]

	cens_coord = 0
	cens_index = 0
	x_axis = []
	y_axis = []

	for i in range(2,8):

		cens_coord,cens_index = k_means(data_2c,i)
		x_axis.append(i)
		dist_s = 0
		distortions = 0
		for z in range(0,i):
			dist_s = 0
			q = 0
			while q < len(data_2c):
				if cens_index[q] == z:
					for s in range(len(data_2c[0])):
						dist_s += (data_2c[q][s] - cens_coord[z][s])**2
					distortions += dist_s
				q += 1
		distortions = distortions/len(data_2c)
		y_axis.append(distortions)




	plt.plot(x_axis,y_axis,"o")
	plt.title("Distortions vs. Number of Centroids")
	plt.xlabel("Number of Centroids (k)")
	plt.ylabel("Distortions")
	plt.show()

