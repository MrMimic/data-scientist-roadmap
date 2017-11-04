#!/usr/bin/env python3
# coding: utf8


""" IMPORTS """
import numpy as np
import matplotlib.pyplot as plt



class Perceptron(object):


	def __init__(self):

		self.l_rate = 0.01
		self.n_epoch = 5
		self.bias = 0
		self.input_data = [[0.2, 0.7], [15.25, 14.37], [0.02, 0.68], [14.55, 16.36], [0.55, 0.36], [0.45, 0.16], [0.45, 0.26], [11.54, 17.226], [12.58, 17.36], [13.95, 15.26]]
		self.expected = [0, 1, 0, 1, 0, 0, 0, 1, 1, 1]

		#~ x = [i[0] for i in self.input_data]
		#~ y = [i[1] for i in self.input_data]
		#~ plt.scatter(x, y)
		#~ plt.show()


	def predict(self, row, weights):

		# Activation threshold function
		activation = 0		
		for i in range(len(row) - 1):
			activation += weights[i] * row[i]
		activation += self.bias
		# Return class
		if activation >= 0.0:
			return 1.0
		else:
			return 0.0


	# Estimate Perceptron weights using stochastic gradient descent
	def train_weights(self):
		
		# Let's initiate weights with small values
		weights = list(np.random.uniform(low = 0, high = 0.1, size = 2))
		global_errors = []
		for epoch in range(self.n_epoch):
			# Keep track of errors for plotting
			epoch_errors = 0.0
			for row, expected in zip(self.input_data, self.expected):
				prediction = self.predict(row, weights)
				error = expected - prediction
				# We keep a track of global errors for plotting
				global_errors.append(abs(error))
				epoch_errors += abs(error)
				# If predicted and expected are different, update weights and bias
				if expected != prediction:
					self.bias = self.bias + self.l_rate * error
					for i in range(len(row)-1):
						weights[i] = weights[i] + self.l_rate * error * row[i]
			print('epoch: {}, lrate: {}, errors: {}'.format(epoch, self.l_rate, epoch_errors))

		plt.plot(global_errors)
		plt.ylim(-1, 2)
		plt.show()

		return weights
 
 
 
 
 
 
if __name__ == '__main__':
	# Calculate weights
	
	neuron = Perceptron()
	
	weights = neuron.train_weights()

	print(neuron.predict([8.23, 9.45], weights))
	print(neuron.predict([0.23, 1.45], weights))






