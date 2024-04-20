# Load the plotly library for interactive visualization
library(plotly)

# Iris is a built-in dataset in R
data <- iris

# Print the dataset
print(data)

# Histogram
column <- data[,1]
hist(column, main = "Histogram of Sepal Length", xlab = "Sepal Length", ylab = "Frequency", col = 'skyblue', border = 'white')

# Box plot
boxplot(column, main = "Boxplot of Sepal Length", ylab = "Sepal Length", col = 'salmon')

# Line chart
X <- data[,1]
Y <- data[,3]
plot_ly(x = X, y = Y, type = "scatter", mode = "markers", marker = list(color = "red")) %>%
  layout(title = "Line Chart of Sepal Length vs. Petal Length",
         xaxis = list(title = "Sepal Length"),
         yaxis = list(title = "Petal Length"))
