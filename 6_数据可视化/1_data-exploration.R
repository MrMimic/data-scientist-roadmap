#####################
# To execute line by line in Rstudio, select it (hightlight)
# Press Ctrl+Enter

# Iris is an array of values examples coming with R.
data <- iris
# This is equal to :
data = iris
# To print it: Sepal.Length Sepal.Width Petal.Length Petal.Width    Species
show(data)

# Histogram
column = data[,1]
hist(column)
# Change parameters :
hist(column, main = "Main title", xlab = "SEPAL LENGTH", ylab = "FREQUENCY", col = 'red', breaks = 10)
hist(column, main = "Main title", xlab = "SEPAL LENGTH", ylab = "FREQUENCY", col = 'red', breaks = 15)

# Box plot
boxplot(column, main = "Main title", ylab = "SEPAL LENGTH", col = 'red')

# Line chart, not very useful here, indeed
X = data[,1]
Y = data[,3]
plot(x = X, y = Y, main = "Main title", xlab = "SEPAL LENGTH", ylab = "PETAL LENGTH", col = 'red')
