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

    sudo pip3 install numpy

#### Utilization
    
    import numpy

## 3_ Exploratory data analysis

The step includes visualization and analysis of data. 

Raw data may possess improper distributions of data which may lead to issues moving forward.

Again, during applications we must also know the distribution of data, for instance, the fact whether the data is linear or spirally distributed.

[Guide to EDA in Python](https://towardsdatascience.com/data-preprocessing-and-interpreting-results-the-heart-of-machine-learning-part-1-eda-49ce99e36655)

##### Libraries in Python 

[Matplotlib](https://matplotlib.org/)

Library used to plot graphs in Python

__Installation__:

    sudo pip3 install matplotlib

__Utilization__:

    import matplotlib.pyplot as plt

[Pandas](https://pandas.pydata.org/)

Library used to large datasets in python

__Installation__:

    sudo pip3 install pandas

__Utilization__:

    import pandas as pd
    
[Seaborn](https://seaborn.pydata.org/)

Yet another Graph Plotting Library in Python.

__Installation__:

    sudo pip3 install seaborn

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

## 5_ Percentiles & outliers

## 6_ Probability theory

## 7_ Bayes theorem

## 8_ Random variables

## 9_ Cumul Dist Fn (CDF)

## 10_ Continuous distributions

## 11_ Skewness

## 12_ ANOVA

## 13_ Prob Den Fn (PDF)

## 14_ Central Limit theorem

## 15_ Monte Carlo method

## 16_ Hypothesis Testing

## 17_ p-Value

## 18_ Chi2 test

## 19_ Estimation

## 20_ Confid Int (CI)

## 21_ MLE

## 22_ Kernel Density estimate

## 23_ Regression

## 24_ Covariance

## 25_ Correlation

## 26_ Pearson coeff

## 27_ Causation

## 28_ Least2-fit

## 29_ Euclidian Distance
