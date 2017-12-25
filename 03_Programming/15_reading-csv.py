#!/usr/bin/python3


''' Example 1 '''
# Import
import csv
# Open file
with open('15_csv-example.csv') as csv_file:
	# Read file with csv library
	read_csv_file = csv.reader(csv_file, delimiter='\t')
	# Parse every row to print it
	for row in read_csv_file:
		print('\'s color is '.join(row))


''' Exemple 2 '''
# Import
import re
# Open file
csv_file = open('15_csv-example.csv', 'r')
for row in csv_file:
	# Get values with regex
	row_data = re.findall('(.*)\t(.*)', str(row))
	# And print
	print('{}\'s color is {}'.format(row_data[0][0], row_data[0][1]))
