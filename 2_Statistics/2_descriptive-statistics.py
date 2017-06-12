#!/usr/bin/python3

# Import
import numpy as np

# Create a dataset
dataset = [12, 52, 45, 65, 78, 11, 12, 54, 56]

# Apply mean/median functions
dataset_mean = np.mean(dataset)
dataset_median = np.median(dataset)

# Print results
print('Mean: {}, median: {}'.format(dataset_mean, dataset_median))
