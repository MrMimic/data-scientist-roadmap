# 2_ Statistics

[Statistics-101 for data noobs](https://medium.com/@debuggermalhotra/statistics-101-for-data-noobs-2e2a0e23a5dc)

## 1_ Pick a dataset

### Datasets repositories

#### Generalists

- [KAGGLE](https://www.kaggle.com/datasets)
- [Google](https://toolbox.google.com/datasetsearch)

#### Medical

- [PMC](https://www.ncbi.nlm.nih.gov/pmc/)

#### Other languages

##### French

- [DATAGOUV](https://www.data.gouv.fr/fr/)

## 2_ Descriptive statistics

### Mean

In probability and statistics, population mean and expected value are used synonymously to refer to one __measure of the central tendency either of a probability distribution or of the random variable__ characterized by that distribution.

For a data set, the terms arithmetic mean, mathematical expectation, and sometimes average are used synonymously to refer to a central value of a discrete set of numbers: specifically, the __sum of the values divided by the number of values__.

![mean_formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/bd2f5fb530fc192e4db7a315777f5bbb5d462c90)

### Median

The median is the value __separating the higher half of a data sample, a population, or a probability distribution, from the lower half__. In simple terms, it may be thought of as the "middle" value of a data set.

### Descriptive statistics in Python

[Numpy](http://www.numpy.org/) is a python library widely used for statistical analysis.

#### Installation

    pip3 install numpy

#### Utilization

    import numpy as np

#### Averages and variances using numpy

|                                        Code                                          |                 Return                  |
|--------------------------------------------------------------------------------------|-----------------------------------------|
|`np.median(a, axis=None, out=None, overwrite_input=False, keepdims=False)`            |   Compute the median along the specified axis |
|`np.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)`  |   Compute the arithmetic mean along the specified axis |
|`np.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>)`            |    Compute the standard deviation along the specified axis. |
|`np.var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>)`            |      Compute the variance along the specified axis  |

#### Code Example

input

```
import numpy as np                                    #import the numpy package
a = np.array([1,2,3,4,5,6,7,8,9])                     #Create a numpy array
print ('median = ' , np.median(a) )                   #Calculate the median of the array
print ('mean = '  , np.mean (a))                      #Calculate the mean of the array
print ('standard deviation = ' , np.std(a) )          #Calculate the standarddeviation of the array
print ('variance = ' , np.var (a) )                   #Calculate the variance of the array
```

output

```
median =  5.0
mean =  5.0
standard deviation =  2.581988897471611
variance =  6.666666666666667
```

you can found more [here](https://numpy.org/doc/stable/reference/routines.statistics.html) on how to apply the Descriptive statistics in Python using numpy package.

## 3_ Exploratory data analysis

The step includes visualization and analysis of data.

Raw data may possess improper distributions of data which may lead to issues moving forward.

Again, during applications we must also know the distribution of data, for instance, the fact whether the data is linear or spirally distributed.

[Guide to EDA in Python](https://towardsdatascience.com/data-preprocessing-and-interpreting-results-the-heart-of-machine-learning-part-1-eda-49ce99e36655)

##### Libraries in Python

[Matplotlib](https://matplotlib.org/)

Library used to plot graphs in Python

__Installation__:

    pip3 install matplotlib

__Utilization__:

    import matplotlib.pyplot as plt

[Pandas](https://pandas.pydata.org/)

Library used to large datasets in python

__Installation__:

    pip3 install pandas

__Utilization__:

    import pandas as pd

[Seaborn](https://seaborn.pydata.org/)

Yet another Graph Plotting Library in Python.

__Installation__:

    pip3 install seaborn

__Utilization__:

    import seaborn as sns

#### PCA

PCA stands for principle component analysis.

We often require to shape of the data distribution as we have seen previously. We need to plot the data for the same.

Data can be Multidimensional, that is, a dataset can have multiple features.

We can plot only two dimensional data, so, for multidimensional data, we project the multidimensional distribution in two dimensions, preserving the principle components of the distribution, in order to get an idea of the actual distribution through the 2D plot.

It is used for dimensionality reduction also. Often it is seen that several features do not significantly contribute any important insight to the data distribution. Such features creates complexity and increase dimensionality of the data. Such features are not considered which results in decrease of the dimensionality of the data.

[Mathematical Explanation](https://medium.com/towards-artificial-intelligence/demystifying-principal-component-analysis-9f13f6f681e6)

[Application in Python](https://towardsdatascience.com/data-preprocessing-and-interpreting-results-the-heart-of-machine-learning-part-2-pca-feature-92f8f6ec8c8)

## 4_ Histograms

Histograms are representation of distribution of numerical data. The procedure consists of binnng the numeric values using range divisions i.e, the entire range in which the data varies is split into several fixed intervals. Count or frequency of occurences of the numbers in the range of the bins are represented.

[Histograms](https://en.wikipedia.org/wiki/Histogram)

![plot](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Example_histogram.png/220px-Example_histogram.png)

In python, __Pandas__,__Matplotlib__,__Seaborn__ can be used to create Histograms.

## 5_ Percentiles & outliers

### Percentiles

Percentiles are numberical measures in statistics, which represents how much or what percentage of data falls below a given number or instance in a numerical data distribution.

For instance, if we say 70 percentile, it represents, 70% of the data in the ditribution are below the given numerical value.

[Percentiles](https://en.wikipedia.org/wiki/Percentile)

### Outliers

Outliers are data points(numerical) which have significant differences with other data points. They differ from majority of points in the distribution. Such points may cause the central measures of distribution, like mean, and median. So, they need to be detected and removed.

[Outliers](https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm)

__Box Plots__ can be used detect Outliers in the data. They can be created using __Seaborn__ library

![Image_Box_Plot](https://miro.medium.com/max/612/1*105IeKBRGtyPyMy3-WQ8hw.png)
  
## 6_ Probability theory

__Probability__ is the likelihood of an event in a Random experiment. For instance, if a coin is tossed, the chance of getting a head is 50% so, probability is 0.5.

__Sample Space__: It is the set of all possible outcomes of a Random Experiment.
__Favourable Outcomes__: The set of outcomes we are looking for in a Random Experiment

__Probability = (Number of Favourable Outcomes) / (Sample Space)__

__Probability theory__ is a branch of mathematics that is associated with the concept of probability.

[Basics of Probability](https://towardsdatascience.com/basic-probability-theory-and-statistics-3105ab637213)

## 7_ Bayes theorem

### Conditional Probability

It is the probability of one event occurring, given that another event has already occurred. So, it gives a sense of relationship between two events and the probabilities of the occurences of those events.

It is given by:

__P( A | B )__ : Probability of occurence of A, after B occured.

The formula is given by:

![formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/74cbddb93db29a62d522cd6ab266531ae295a0fb)

So, P(A|B) is equal to Probablity of occurence of A and B, divided by Probability of occurence of B.

[Guide to Conditional Probability](https://en.wikipedia.org/wiki/Conditional_probability)

### Bayes Theorem

Bayes theorem provides a way to calculate conditional probability. Bayes theorem is widely used in machine learning most in Bayesian Classifiers.  

According to Bayes theorem the probability of A, given that B has already occurred is given by Probability of A multiplied by the probability of B given A has already occurred divided by the probability of B.

__P(A|B) =  P(A).P(B|A) / P(B)__

[Guide to Bayes Theorem](https://machinelearningmastery.com/bayes-theorem-for-machine-learning/)

## 8_ Random variables

Random variable are the numeric outcome of an experiment or random events. They are normally a set of values.

There are two main types of Random Variables:

__Discrete Random Variables__: Such variables take only a finite number of distinct values

__Continous Random Variables__: Such variables can take an infinite number of possible values.

## 9_ Cumul Dist Fn (CDF)

In probability theory and statistics, the cumulative distribution function (CDF) of a real-valued random variable __X__, or just distribution function of __X__, evaluated at __x__, is the probability that __X__ will take a value less than or equal to __x__.

The cumulative distribution function of a real-valued random variable X is the function given by:

![CDF](https://wikimedia.org/api/rest_v1/media/math/render/svg/f81c05aba576a12b4e05ee3f4cba709dd16139c7)

Resource:

[Wikipedia](https://en.wikipedia.org/wiki/Cumulative_distribution_function)

## 10_ Continuous distributions

A continuous distribution describes the probabilities of the possible values of a continuous random variable. A continuous random variable is a random variable with a set of possible values (known as the range) that is infinite and uncountable.

## 11_ Skewness

Skewness is the measure of assymetry in the data distribution or a random variable distribution about its mean.

Skewness can be positive, negative or zero.

![skewed image](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Negative_and_positive_skew_diagrams_%28English%29.svg/446px-Negative_and_positive_skew_diagrams_%28English%29.svg.png)

__Negative skew__: Distribution Concentrated in the right, left tail is longer.

__Positive skew__: Distribution Concentrated in the left, right tail is longer.

Variation of central tendency measures are shown below.

![cet](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Relationship_between_mean_and_median_under_different_skewness.png/434px-Relationship_between_mean_and_median_under_different_skewness.png)

Data Distribution are often Skewed which may cause trouble during processing the data. __Skewed Distribution can be converted to Symmetric Distribution, taking Log of the distribution__.

##### Skew Distribution

![Skew](https://miro.medium.com/max/379/1*PLSczKIQRc8ZtlvHED-6mQ.png)

##### Log of the Skew Distribution

![log](https://miro.medium.com/max/376/1*4GFayBYKIiqAcyI69wIFzA.png)

[Guide to Skewness](https://en.wikipedia.org/wiki/Skewness)

## 12_ ANOVA

ANOVA stands for __analysis of variance__.

It is used to compare among groups of data distributions.

Often we are provided with huge data. They are too huge to work with. The total data is called the __Population__.

In order to work with them, we pick random smaller groups of data. They are called __Samples__.

ANOVA is used to compare the variance among these groups or samples.

Variance of  group is given by:

![var](https://miro.medium.com/max/446/1*yzAMFVIEFysMKwuT0YHrZw.png)

The differences in the collected samples are observed using the differences between the means of the groups. We often use the __t-test__ to compare the means and also to check if the samples belong to the same population,

Now, t-test can only be possible among two groups. But, often we get more groups or samples.

If we try to use t-test for more than two groups we have to perform t-tests multiple times, once for each pair. This is where ANOVA is used.

ANOVA has two components:

__1.Variation within each group__

__2.Variation between groups__

It works on a ratio called the  __F-Ratio__

It is given by:

![F-ratio](https://miro.medium.com/max/491/1*I5dSwtUICySQ5xvKmq6M8A.png)

F ratio shows how much of the total variation comes from the variation between groups and how much comes from the variation within groups. If much of the variation comes from the variation between groups, it is more likely that the mean of groups are different. However, if most of the variation comes from the variation within groups, then we can conclude the elements in a group are different rather than entire groups. The larger the F ratio, the more likely that the groups have different means.

Resources:

[Defnition](https://statistics.laerd.com/statistical-guides/one-way-anova-statistical-guide.php)

[GUIDE 1](https://towardsdatascience.com/anova-analysis-of-variance-explained-b48fee6380af)

[Details](https://medium.com/@StepUpAnalytics/anova-one-way-vs-two-way-6b3ff87d3a94)

## 13_ Prob Den Fn (PDF)

It stands for probability density function.

__In probability theory, a probability density function (PDF), or density of a continuous random variable, is a function whose value at any given sample (or point) in the sample space (the set of possible values taken by the random variable) can be interpreted as providing a relative likelihood that the value of the random variable would equal that sample.__

The probability density function (PDF) P(x) of a continuous distribution is defined as the derivative of the (cumulative) distribution function D(x).

It is given by the integral of the function over a given range.

![PDF](https://wikimedia.org/api/rest_v1/media/math/render/svg/45fd7691b5fbd323f64834d8e5b8d4f54c73a6f8)

## 14_ Central Limit theorem

## 15_ Monte Carlo method

## 16_ Hypothesis Testing

### Types of curves

We need to know about two distribution curves first.

Distribution curves reflect the probabilty of finding an instance or a sample of a population at a certain value of the distribution.

__Normal Distribution__

![normal distribution](https://sciences.usca.edu/biology/zelmer/305/norm/stanorm.jpg)

The normal distribution represents how the data is distributed. In this case, most of the data samples in the distribution are scattered at and around the mean of the distribution. A few instances are scattered or present at the long tail ends of the distribution.

Few points about Normal Distributions are:

1. The curve is always Bell-shaped. This is because most of the data is found around the mean, so the proababilty of finding a sample at the mean or central value is more.

2. The curve is symmetric

3. The area under the curve is always 1. This is because all the points of the distribution must be present under the curve

4. For Normal Distribution, Mean and Median lie on the same line in the distribution.

__Standard Normal Distribution__

This type of distribution are normal distributions which following conditions.

1. Mean of the distribution is 0

2. The Standard Deviation of the distribution is equal to 1.

The idea of Hypothesis Testing works completely on the data distributions.

### Hypothesis Testing

Hypothesis testing is a statistical method that is used in making statistical decisions using experimental data. Hypothesis Testing is basically an assumption that we make about the population parameter.

For example, say, we take the hypothesis that boys in a class are taller than girls.

The above statement is just an assumption on the population of the class.

__Hypothesis__ is just an assumptive proposal or statement made on the basis of observations made on a set of information or data.

We initially propose two mutually exclusive statements based on the population of the sample data.

The initial one is called __NULL HYPOTHESIS__. It is denoted by H0.

The second one is called __ALTERNATE HYPOTHESIS__. It is denoted by H1 or Ha. It is used as a contrary to Null Hypothesis.

Based on the instances of the population we accept or reject the NULL Hypothesis and correspondingly we reject or accept the ALTERNATE Hypothesis.

#### Level of Significance

It is the degree which we consider to decide whether to accept or reject the NULL hypothesis. When we consider a hypothesis on a population, it is not the case that 100% or all instances of the population abides the assumption, so we decide a __level of significance as a cutoff degree, i.e, if our level of significance is 5%, and (100-5)% = 95% of the data abides by the assumption, we accept the Hypothesis.__

__It is said with 95% confidence, the hypothesis is accepted__

![curve](https://i.stack.imgur.com/d8iHd.png)

The non-reject region is called __acceptance region or beta region__. The rejection regions are called __critical or alpha regions__. __alpha__ denotes the __level of significance__.

If level of significance is 5%. the two alpha regions have (2.5+2.5)% of the population and the beta region has the 95%.

The acceptance and rejection gives rise to two kinds of errors:

__Type-I Error:__ NULL Hypothesis is true, but wrongly Rejected.

__Type-II Error:__ NULL Hypothesis if false but is wrongly accepted.

![hypothesis](https://microbenotes.com/wp-content/uploads/2020/07/Graphical-representation-of-type-1-and-type-2-errors.jpg)

### Tests for Hypothesis

__One Tailed Test__:

![One-tailed](https://prwatech.in/blog/wp-content/uploads/2019/07/onetailtest.png)

This is a test for Hypothesis, where the rejection region is only one side of the sampling distribution. The rejection region may be in right tail end or in the left tail end.

The idea is if we say our level of significance is 5% and we consider a hypothesis "Hieght of Boys in a class is <=6 ft". We consider the hypothesis true if atmost 5% of our population are more than 6 feet. So, this will be one-tailed as the test condition only restricts one tail end, the end with hieght > 6ft.

![Two Tailed](https://i0.wp.com/www.real-statistics.com/wp-content/uploads/2012/11/two-tailed-significance-testing.png)

In this case, the rejection region extends at both tail ends of the distribution.

The idea is if we say our level of significance is 5% and we consider a hypothesis "Hieght of Boys in a class is !=6 ft".

Here, we can accept the NULL hyposthesis iff atmost 5% of the population is less than or greater than 6 feet. So, it is evident that the crirtical region will be at both tail ends and the region is 5% / 2 = 2.5% at both ends of the distribution.

## 17_ p-Value

Before we jump into P-values we need to look at another important topic in the context: Z-test.

### Z-test

We need to know two terms: __Population and Sample.__

__Population__ describes the entire available data distributed. So, it refers to all records provided in the dataset.

__Sample__ is said to be a group of data points randomly picked from a population or a given distribution. The size of the sample can be any number of data points, given by __sample size.__

__Z-test__ is simply used to determine if a given sample distribution belongs to a given population.

Now,for Z-test we have to use __Standard Normal Form__ for the standardized comparison measures.

![std1](https://miro.medium.com/max/700/1*VYCN5b-Zubr4rrc9k37SAg.png)

As we already have seen, standard normal form is a normal form with mean=0 and standard deviation=1.

The __Standard Deviation__ is a measure of how much differently the points are distributed around the mean.

![std2](https://miro.medium.com/max/640/1*kzFQaZ08dTjlPq1zrcJXgg.png)

It states that approximately 68% , 95% and 99.7% of the data lies within 1, 2 and 3 standard deviations of a normal distribution respectively.

Now, to convert the normal distribution to standard normal distribution we need a standard score called Z-Score.
It is given by:

![Z-score](https://miro.medium.com/max/125/1*X--kDNyurDEo2zKbSDDf-w.png)

x = value that we want to standardize

µ = mean of the distribution of x

σ = standard deviation of the distribution of x

We need to know another concept __Central Limit Theorem__.

##### Central Limit Theorem

_The theorem states that the mean of the sampling distribution of the sample means is equal to the population mean irrespective if the distribution of population where sample size is greater than 30._

And

_The sampling distribution of sampling mean will also follow the normal distribution._

So, it states, if we pick several samples from a distribution with the size above 30, and pick the static sample means and use the sample means to create a distribution, the mean of the newly created sampling distribution is equal to the original population mean.

According to the theorem, if we draw samples of size N, from a population with population mean μ and population standard deviation σ, the condition stands:

![std3](https://miro.medium.com/max/121/0*VPW964abYGyevE3h.png)

i.e, mean of the distribution of sample means is equal to the sample means.

The standard deviation of the sample means is give by:

![std4](https://miro.medium.com/max/220/0*EMx4C_A9Efsd6Ef6.png)

The above term is also called standard error.

We use the theory discussed above for Z-test. If the sample mean lies close to the population mean, we say that the sample belongs to the population and if it lies at a distance from the population mean, we say the sample is taken from a different population.

To do this we use a formula and check if the z statistic is greater than or less than 1.96 (considering two tailed test, level of significance = 5%)

![los](https://miro.medium.com/max/424/0*C9XaCIUWoJaBSMeZ.gif)

![std5](https://miro.medium.com/max/137/1*DRiPmBtjK4wmidq9Ha440Q.png)

 The above formula gives Z-static

z = z statistic

X̄ = sample mean

μ = population mean

σ = population standard deviation

n = sample size

Now, as the Z-score is used to standardize the distribution, it gives us an idea how the data is distributed overall.

### P-values

It is used to check if the results are statistically significant based on the significance level.  

Say, we perform an experiment and collect observations or data. Now, we make a hypothesis (NULL hypothesis) primary, and a second hypothesis, contradictory to the first one called the alternative hypothesis.

Then we decide a level of significance which serve as a threshold for our null hypothesis. The P value actually gives the probability of the statement. Say, the p-value of our alternative hypothesis is 0.02, it means the probability of alternate hypothesis happenning is 2%.

Now, the level of significance into play to decide if we can allow 2% or p-value of 0.02. It can be said as a level of endurance of the null hypothesis. If our level of significance is 5% using a two tailed test, we can allow 2.5% on both ends of the distribution, we accept the NULL hypothesis, as level of significance > p-value of alternate hypothesis.

But if the p-value is greater than level of significance, we tell that the result is __statistically significant, and we reject NULL hypothesis.__ .

Resources:

1. <https://medium.com/analytics-vidhya/everything-you-should-know-about-p-value-from-scratch-for-data-science-f3c0bfa3c4cc>

2. <https://towardsdatascience.com/p-values-explained-by-data-scientist-f40a746cfc8>

3.<https://medium.com/analytics-vidhya/z-test-demystified-f745c57c324c>

## 18_ Chi2 test

Chi2 test is extensively used in data science and machine learning problems for feature selection.

A chi-square test is used in statistics to test the independence of two events. So, it is used to check for independence of features used. Often dependent features are used which do not convey a lot of information but adds dimensionality to a feature space.

It is one of the most common ways to examine relationships between two or more categorical variables.

It involves calculating a number, called the chi-square statistic - χ2. Which follows a chi-square distribution.

It is given as the summation of the difference of the expected values and observed value divided by the observed value.

![Chi2](https://miro.medium.com/max/266/1*S8rfFkmLhDbOz4RGNwuz6g.png)

Resources:

[Definitions](investopedia.com/terms/c/chi-square-statistic.asp)

[Guide 1](https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223)

[Guide 2](https://medium.com/swlh/what-is-chi-square-test-how-does-it-work-3b7f22c03b01)

[Example of Operation](https://medium.com/@kuldeepnpatel/chi-square-test-of-independence-bafd14028250)

## 19_ Estimation

## 20_ Confid Int (CI)

## 21_ MLE

## 22_ Kernel Density estimate

In statistics, kernel density estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable. Kernel density estimation is a fundamental data smoothing problem where inferences about the population are made, based on a finite data sample.

Kernel Density estimate can be regarded as another way to represent the probability distribution.

![KDE1](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Kernel_density.svg/250px-Kernel_density.svg.png)

It consists of choosing a kernel function. There are mostly three used.

1. Gaussian

2. Box

3. Tri

The kernel function depicts the probability of finding a data point. So, it is highest at the centre and decreases as we move away from the point.

We assign a kernel function over all the data points and finally calculate the density of the functions, to get the density estimate of the distibuted data points. It practically adds up the Kernel function values at a particular point on the axis. It is as shown below.

![KDE 2](https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Comparison_of_1D_histogram_and_KDE.png/500px-Comparison_of_1D_histogram_and_KDE.png)

Now, the kernel function is given by:

![kde3](https://wikimedia.org/api/rest_v1/media/math/render/svg/f3b09505158fb06033aabf9b0116c8c07a68bf31)

where K is the kernel — a non-negative function — and h > 0 is a smoothing parameter called the bandwidth.

The 'h' or the bandwidth is the parameter, on which the curve varies.

![kde4](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Comparison_of_1D_bandwidth_selectors.png/220px-Comparison_of_1D_bandwidth_selectors.png)

Kernel density estimate (KDE) with different bandwidths of a random sample of 100 points from a standard normal distribution. Grey: true density (standard normal). Red: KDE with h=0.05. Black: KDE with h=0.337. Green: KDE with h=2.

Resources:

[Basics](https://www.youtube.com/watch?v=x5zLaWT5KPs)

[Advanced](https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html)

## 23_ Regression

Regression tasks deal with predicting the value of a __dependent variable__ from a set of __independent variables.__

Say, we want to predict the price of a car. So, it becomes a dependent variable say Y, and the features like engine capacity, top speed, class, and company become the independent variables, which helps to frame the equation to obtain the price.

If there is one feature say x. If the dependent variable y is linearly dependent on x, then it can be given by __y=mx+c__, where the m is the coefficient of the independent in the equation, c is the intercept or bias.

The image shows the types of regression

![types](https://miro.medium.com/max/2001/1*dSFn-uIYDhDfdaG5GXlB3A.png)

[Guide to Regression](https://towardsdatascience.com/a-deep-dive-into-the-concept-of-regression-fb912d427a2e)

## 24_ Covariance

### Variance

The variance is a measure of how dispersed or spread out the set is. If it is said that the variance is zero, it means all the elements in the dataset are same. If the variance is low, it means the data are slightly dissimilar. If the variance is very high, it means the data in the dataset are largely dissimilar.

Mathematically, it is a measure of how far each value in the data set is from the mean.

Variance (sigma^2) is given by summation of the square of distances of each point from the mean, divided by the number of points

![formula var](https://cdn.sciencebuddies.org/Files/474/9/DefVarEqn.jpg)

### Covariance

Covariance gives us an idea about the degree of association between two considered random variables. Now, we know random variables create distributions. Distribution are a set of values or data points which the variable takes and we can easily represent as vectors in the vector space.

For vectors covariance is defined as the dot product of two vectors. The value of covariance can vary from positive infinity to negative infinity. If the two distributions or vectors grow in the same direction the covariance is positive and vice versa. The Sign gives the direction of variation and the Magnitude gives the amount of variation.  

Covariance is given by:

![cov_form](https://cdn.corporatefinanceinstitute.com/assets/covariance1.png)

where Xi and Yi denotes the i-th point of the two distributions and X-bar and Y-bar represent the mean values of both the distributions, and n represents the number of values or data points in the distribution.

## 25_ Correlation

Covariance measures the total relation of the variables namely both direction and magnitude. Correlation is a scaled measure of covariance. It is dimensionless and independent of scale. It just shows the strength of variation for both the variables.

Mathematically, if we represent the distribution using vectors, correlation is said to be the cosine angle between the vectors. The value of correlation varies from +1 to -1. +1 is said to be a strong positive correlation and -1 is said to be a strong negative correlation. 0 implies no correlation, or the two variables are independent of each other.

Correlation is given by:

![corr](https://cdn.corporatefinanceinstitute.com/assets/covariance3.png)

Where:

ρ(X,Y) – the correlation between the variables X and Y

Cov(X,Y) – the covariance between the variables X and Y

σX – the standard deviation of the X-variable

σY – the standard deviation of the Y-variable

Standard deviation is given by square roo of variance.

## 26_ Pearson coeff

## 27_ Causation

## 28_ Least2-fit

## 29_ Euclidian Distance

__Eucladian Distance is the most used and standard measure for the distance between two points.__

It is given as the square root of sum of squares of the difference between coordinates of two points.

__The Euclidean distance between two points in Euclidean space is a number, the length of a line segment between the two points. It can be calculated from the Cartesian coordinates of the points using the Pythagorean theorem, and is occasionally called the Pythagorean distance.__

__In the Euclidean plane, let point p have Cartesian coordinates (p_{1},p_{2}) and let point q have coordinates (q_{1},q_{2}). Then the distance between p and q is given by:__

![eucladian](https://wikimedia.org/api/rest_v1/media/math/render/svg/9c0157084fd89f5f3d462efeedc47d3d7aa0b773)
