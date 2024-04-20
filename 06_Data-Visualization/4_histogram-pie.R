# Set up the layout for two plots on the same window
par(mfrow = c(1,2))

# Histogram
data <- iris
hist(data[,2], main = "Histogram of Sepal Width", xlab = "Sepal Width", ylab = "Frequency", col = "skyblue", border = "white")
box()  # Add a box around the histogram plot

# Pie chart
classes <- summary(data[,5])
pie(classes, main = "Distribution of Iris Species", col = rainbow(length(classes)))  # Add colors to the pie chart
legend("topright", legend = rownames(classes), fill = rainbow(length(classes)), cex = 0.8)  # Add a legend
