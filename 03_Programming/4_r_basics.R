## PACKAGES

# To install the package "foobar"
install.packages("foobar")
# And load it
library(foobar)
# Package documentation
?foobar
# or
help("foobar")

## DATASET

# Load in environment
data(iris)
# Import data already loaded in R into the variable "data"
data <- iris
# The same as
data = iris
# To read a CSV
data <- read.csv('path/to/the/file', sep = ',')
# description
str(iris)
# statistical summary
summary(iris)
# type of object
class(iris) # data frame
# names of variables (columns)
names(iris)
# num rows
nrow(iris)
# num columns
ncol(iris)
# dimension
dim(iris)
# Select the 2nd column
data[,2]
# And the 3rd row
data[3,]
# Mean of the 2nd column
mean(data[,2])
# Histogram of the 3rd column
hist(data[,3])

## DATA WRANGLING

# access variables of data frame
iris$Sepal.Length
iris$Species
# or
iris["Species"]
# row subsetting w.r.t column 
iris[, "Petal.Width"]
# column subsetting w.r.t row
iris[1:10, ]
