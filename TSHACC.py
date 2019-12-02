import numpy as np
import copy as cp
import scipy
import time
import random
import sklearn
import sys

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


# Solves all pair shortest path via Floyd Warshall Algorithm 
def floyd_warshall(graph): 

	""" dist[][] will be the output matrix that will finally 
		have the shortest distances between every pair of vertices """
	""" initializing the solution matrix same as input graph matrix 
	OR we can say that the initial values of shortest distances 
	are based on shortest paths considering no 
	intermediate vertices """
	dist = cp.deepcopy(graph)

	V = np.shape(graph)[0] 
	
	""" Add all vertices one by one to the set of intermediate 
	vertices. 
	---> Before start of an iteration, we have shortest distances 
	between all pairs of vertices such that the shortest 
	distances consider only the vertices in the set 
	{0, 1, 2, .. k-1} as intermediate vertices. 
	----> After the end of a iteration, vertex no. k is 
	added to the set of intermediate vertices and the 
	set becomes {0, 1, 2, .. k} 
	"""
	for k in range(V): 

		# pick all vertices as source one by one 
		for i in range(V): 

			# Pick all vertices as destination for the 
			# above picked source 
			for j in range(V): 

				# If vertex k is on the shortest path from 
				# i to j, then update the value of dist[i][j] 
				dist[i,j] = min(dist[i,j], dist[i,k]+ dist[k,j]) 

	return dist


def constrained_space_dist(data, const):

	#Compute Euclidean distances
	D = sklearn.metrics.pairwise_distances(data, Y=None, metric='euclidean')

	#Set distances between ml-linked instances to 0
	D[const == 1] = 0

	#Recompute distances using all pairs shortest paths
	D = floyd_warshall(D)

	#Set distances between cl-linked instances to a value over the maximum
	max_D = np.max(D)
	D[const == -1] = max_D * 1.5

	return D


def RCPD(data, const, lamb, beta, k_thresh, norm_thresh, max_it):

	X = np.transpose(data)
	n_features = X.shape[0]
	n_instances = X.shape[1]
	Z = np.ones((n_instances, n_instances))
	Z = np.random.randint(low = 1, high = 99, size = (n_instances, n_instances)) #Filas = instancias, Columnas = clusters
	Z = np.asmatrix(Z, dtype = np.float)
	Z = Z / np.sum(Z, axis = 1)

	for i in range(Z.shape[0]):

		Z[i, :] += Z[i,i]/(Z.shape[0]-1)

	np.fill_diagonal(Z, 0.0)
	old_Z = np.zeros((n_instances, n_instances))
	it = 0

	#Compute distances in the dataset to get P

	P = constrained_space_dist(data, const)**2
	# P = sklearn.metrics.pairwise_distances(data, Y=None, metric='sqeuclidean')

	#Iterate over matrix Z until convergence

	while np.linalg.norm(Z - old_Z, ord = "fro") > norm_thresh and it < max_it:

		# print(np.linalg.norm(Z - old_Z, ord = "fro"))

		for i in range(n_instances):

			#Extract i-th column (instance) from matrix X (data matrix)

			x = np.asmatrix(X[:, i]).transpose()

			#Extract i-th row from matrix Z

			z_t = np.asmatrix(Z[i, :])

			#Extract i-th column from euclidean distance matrix P

			p = np.asarray(P[:, i])

			#Compute matrix X_1

			X_1 = X - (np.matmul(X, Z) - np.matmul(x, z_t))

			#Compute vector v

			v = np.asarray(np.matmul(X_1.transpose(), x) / (np.matmul(x.transpose(), x) +  beta)).reshape(-1)

			#Update values of i-th row of matrix Z
			old_Z = cp.deepcopy(Z)
			for k in range(n_instances):

				if k == i:

					Z[i,k] = 0

				elif (np.abs(v[k]) - (lamb*p[k])/2) > 0:

					Z[i,k] = np.sign(v[k]) * (np.abs(v[k]) - ((lamb*p[k])/2))

				else:

					Z[i,k] = 0


		it += 1

	#Apply a hard thresholding operator to Z to select the k largest entries in each column

	for i in range(n_instances):

		row = np.array(Z[:, i].transpose().flatten())[0]
		below_threshold_value = np.sort(row)[::-1][k_thresh-1]
		Z[:,i][Z[:,i] < below_threshold_value] = 0


	W = np.abs(Z)

	W = (W + W.transpose())/2

	return sklearn.preprocessing.normalize(W, norm='l2', axis = 0)


def get_merge_infeasibility(ml_const, cl_const, partition, c1, c2):

	cluster_labels, cluster_count = np.unique(partition, return_counts = True)
	l1 = cluster_labels[c1]
	l2 = cluster_labels[c2]

	new_partition = cp.deepcopy(partition)
	new_partition[partition == l1] = l2
	infeasibility = 0

	# Calculamos el numero de restricciones must-link que no se satisfacen
	for c in range(np.shape(ml_const)[0]):

		if new_partition[ml_const[c][0]] != new_partition[ml_const[c][1]]:
			infeasibility += 1

	# Calculamos el numero de restricciones cannot-link que no se satisfacen
	for c in range(np.shape(cl_const)[0]):

		if new_partition[cl_const[c][0]] == new_partition[cl_const[c][1]]:
			infeasibility += 1

	return infeasibility


def get_aff(W, partition, const):

	total_const = np.count_nonzero(const)
	#Get cluster labels and number of instances per cluster
	cluster_labels, cluster_count = np.unique(partition, return_counts = True)
	#Initialize Affinity matrix as empty
	A = np.zeros([len(cluster_labels), len(cluster_labels)])
	A[A == 0] = -1

	#Assign each entry of Affinity matrix
	for i in range(len(cluster_labels)):

		for j in range(i+1, len(cluster_labels)):

			#Get number of instances for clusters i and j
			card_i = cluster_count[cluster_labels == cluster_labels[i]][0]
			card_j = cluster_count[cluster_labels == cluster_labels[j]][0]

			#Get W subamtrices for cluster i and j
			cluster_i = np.where(partition == cluster_labels[i])
			cluster_j = np.where(partition == cluster_labels[j])

			W_ij = np.asmatrix(W[cluster_i[0],:])[:,cluster_j[0]]
			W_ji = np.asmatrix(W[cluster_j[0],:])[:,cluster_i[0]]

			#Compute first term of the affinity calculation
			v_11 = np.matrix([1/ card_i**2] * card_i)
			v_12 = np.ones((card_i, 1))

			first_term = np.matmul(v_11,W_ij)
			first_term = np.matmul(first_term, W_ji)
			first_term = np.matmul(first_term, v_12)

			#Compute second term of the affinity calculation
			v_21 = np.matrix([1/ card_j**2] * card_j)
			v_22 = np.ones((card_j, 1))

			second_term = np.matmul(v_21,W_ji)
			second_term = np.matmul(second_term, W_ij)
			second_term = np.matmul(second_term, v_22)

		A[i,j] = first_term + second_term

	return A

def get_distances(data, partition, ml_const, cl_const):

	total_const = len(ml_const) + len(cl_const)
	cluster_labels, cluster_count = np.unique(partition, return_counts = True)
	#Initialize Affinity matrix as empty
	clustered_data = np.zeros((len(cluster_labels), data.shape[1]))

	for i in range(len(cluster_labels)):

		cluster_idices = np.where(partition == cluster_labels[i])
		cluster = np.asmatrix(data[cluster_idices[0],:])

		clustered_data[i, :] = np.mean(cluster, axis = 0)

	distances = sklearn.metrics.pairwise_distances(clustered_data, Y=None, metric='euclidean')

	for i in range(len(cluster_labels)):

		for j in range(i+1, len(cluster_labels)):

			infs = get_merge_infeasibility(ml_const, cl_const, partition, i, j)

			penalty_distance = distances[i,j] + infs * 1000

			distances[i,j] = penalty_distance
			distances[j,i] = penalty_distance

	np.fill_diagonal(distances, sys.float_info.max)

	return distances


def RCPD_AC(data, const, ml_const, cl_const, n_clusters, lamb, beta, k_thresh, norm_thresh, max_it):

	#Obtain similarity matrix using the RCPD method
	W = RCPD(data, const, lamb, beta, k_thresh, norm_thresh, max_it)

	#Initialize partition of the dataset. Each instances has its own cluster
	partition = np.array(range(0, data.shape[0]))

	#Get the initial number of clusters
	number_clusters = data.shape[0]

	while number_clusters > n_clusters:

		# print(number_clusters)

		#Compute affinities
		A = get_aff(W, partition, const)
		#Get cluster labels
		cluster_labels, cluster_count = np.unique(partition, return_counts = True)

		#Get the two clusters with the greatest affinity to merge them
		to_merge = np.unravel_index(np.argmax(A, axis=None), A.shape)

		#If maximum affnity is 0, then merge the two closest clusters
		if A[to_merge[0], to_merge[1]] <= 0.0:
			# print("Entrando")
			D = get_distances(data, partition, ml_const, cl_const)
			to_merge = np.unravel_index(np.argmin(D, axis=None), D.shape)

		#Merge both clusters
		partition[partition == cluster_labels[to_merge[1]]] = cluster_labels[to_merge[0]]

		number_clusters = len(np.unique(partition, return_counts = True)[0])

	#Return the n_cluster-partition of the dataset
	return partition




