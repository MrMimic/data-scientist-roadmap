# Import data already loaded in R into the variable "data"
data <- iris
# The same as
data = iris
# To read a CSV
data <- read.csv('path/to/the/file', sep = ',')
# Select the 2nd column
data[,2]
# And the 3rd line
data[3,]
# Mean of the 2nd column
mean(data[,2])
# Histogram of the 3rd column
hist(data[,3])
# To install the package "foobar"
install.packages("foobar")
# And load it
library(foobar)
