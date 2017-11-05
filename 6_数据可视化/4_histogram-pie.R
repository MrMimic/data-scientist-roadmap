# Two plot on the same window
par(mfrow = c(1,2))
# Histogram
data <- iris
hist(data[,2], main = "histogram about sepal width", xlab = "sepal width", ylab = "Frequency")
# Pie chart
classes <- summary(data[,5])
pie(classes, main = "Iris species")
