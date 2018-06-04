import numpy as np

def label2onehot(Y, num_class):
	'''
	This function aim to convert label to one hot matrix.

	Arguments:
	Y -- label vector (m, 1)
	num_class -- (int) number of class

	Returns:
	Y_onehot -- one hot matrix
	'''

	m = Y.shape[0]
	Y_onehot = np.zeros((m, num_class))

	for i in range(m):
		Y_onehot[i, Y[i]-1] = 1

	return Y_onehot
