#!/usr/bin/python3

""" Example 1 """

# Import
import pandas

# Open file
data = pandas.read_csv("15_csv-example.csv", delimiter="\t")
# Print data
for index, row in data.iterrows():
    print(f"{row['Name']}'s color is {row['Color']}")

""" Exemple 2 """

# Import
import csv

# Open file
with open("15_csv-example.csv", encoding="utf-8") as csv_file:
    # Read file with csv library
    read_csv_file = csv.reader(csv_file, delimiter="\t")
    # Parse every row to print it
    for row in read_csv_file:
        print("'s color is ".join(row))
