#!/usr/bin/python3


import numpy as np


# Generate a list of 4 list of 5 random numbers each
list_of_lists = []
# Loop the action 4 times
for i in range(4):
	# Generate a list of 5 numbers between 0 and 100 and add this list to [list_of_lists]
	list_of_lists.append(np.random.randint(low = 0, high = 100, size = 5))
# Convert list_of_lists into numpy matrix
matrix = np.matrix(list_of_lists)
print('Here is your matrix:\n{}\n'.format(matrix))

# Addition
new_matrix = np.sum([matrix, 5])
print('Here is your matrix with addition +5:\n{}\n'.format(new_matrix))

# Multiplication
new_matrix = np.multiply(matrix, matrix)
print('Here is your matrix multiplied by itself:\n{}\n'.format(new_matrix))

# Transposition
new_matrix = np.transpose(matrix)
print('Here is your matrix transposed:\n{}\n'.format(new_matrix))
