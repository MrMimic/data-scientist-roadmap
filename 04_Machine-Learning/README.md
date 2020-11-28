# 4_ Machine learning

## 1_ What is ML ?

### Definition

Machine Learning is part of the Artificial Intelligences study. It concerns the conception, devloppement and implementation of sophisticated methods, allowing a machine to achieve really hard tasks, nearly impossible to solve with classic algorithms.

Machine learning mostly consists of three algorithms:

![ml](https://miro.medium.com/max/561/0*qlvUmkmkeefqe_Mk)

### Utilisation examples

* Computer vision
* Search engines
* Financial analysis
* Documents classification
* Music generation
* Robotics ...

## 2_ Numerical var

Variables which can take continous integer or real values. They can take infinite values.

These types of variables are mostly used for features which involves measurements. For example, hieghts of all students in a class.

## 3_ Categorical var

Variables that take finite discrete values. They take a fixed set of values, in order to classify a data item.

They act like assigned labels. For example: Labelling the students of a class according to gender: 'Male' and 'Female'

## 4_ Supervised learning

Supervised learning is the machine learning task of inferring a function from __labeled training data__. 

The training data consist of a __set of training examples__. 

In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). 

A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. 

In other words:

Supervised Learning learns from a set of labeled examples. From the instances and the labels, supervised learning models try to find the correlation among the features, used to describe an instance, and learn how each feature contributes to the label corresponding to an instance. On receiving an unseen instance, the goal of supervised learning is to label the instance based on its feature correctly.

__An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances__.

## 5_ Unsupervised learning

Unsupervised machine learning is the machine learning task of inferring a function to describe hidden structure __from "unlabeled" data__ (a classification or categorization is not included in the observations). 

Since the examples given to the learner are unlabeled, there is no evaluation of the accuracy of the structure that is output by the relevant algorithm—which is one way of distinguishing unsupervised learning from supervised learning and reinforcement learning.

Unsupervised learning deals with data instances only. This approach tries to group data and form clusters based on the similarity of features. If two instances have similar features and placed in close proximity in feature space, there are high chances the two instances will belong to the same cluster. On getting an unseen instance, the algorithm will try to find, to which cluster the instance should belong based on its feature.

Resource:

[Guide to unsupervised learning](https://towardsdatascience.com/a-dive-into-unsupervised-learning-bf1d6b5f02a7)

## 6_ Concepts, inputs and attributes

A machine learning problem takes in the features of a dataset as input.

For supervised learning, the model trains on the data and then it is ready to perform. So, for supervised learning, apart from the features we also need to input  the corresponding labels of the data points to let the model train on them.

For unsupervised learning, the models simply perform by just citing complex relations among data items and grouping them accordingly. So, unsupervised learning do not need a labelled dataset. The input is only the feature section of the dataset.

## 7_ Training and test data

If we train a supervised machine learning model using a dataset, the model captures the dependencies of that particular data set very deeply. So, the model will always perform well on the data and it won't be proper measure of how well the model performs. 

To know how well the model performs, we must train and test the model on different datasets. The dataset we train the model on is called Training set, and the dataset we test the model on is called the test set.

We normally split the provided dataset to create the training and test set. The ratio of splitting is majorly: 3:7 or 2:8 depending on the data, larger being the trining data.

#### sklearn.model_selection.train_test_split is used for splitting the data.

Syntax:

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  
[Sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

## 8_ Classifiers

Classification is the most important and most common machine learning problem. Classification problems can be both suprvised and unsupervised problems.

The classification problems involve labelling data points to belong to a particular class based on the feature set corresponding to the particluar data point.

Classification tasks can be performed using both machine learning and deep learning techniques.

Machine learning classification techniques involve: Logistic Regressions, SVMs, and Classification trees. The models used to perform the classification are called classifiers.

## 9_ Prediction

The output generated by a machine learning models for a particuolar problem is called its prediction. 

There are majorly two kinds of predictions corresponding to two types of problen: 

1. Classification

2. Regression

In classiication, the prediction is mostly a class or label, to which a data points belong

In regression, the prediction is a number, a continous a numeric value, because regression problems deal with predicting the value. For example, predicting the price of a house.

## 10_ Lift

## 11_ Overfitting

Often we train our model so much or make our model so complex that our model fits too tghtly with the training data.

The training data often contains outliers or represents misleading patterns in the data. Fitting the training data with such irregularities to deeply cause the model to lose its generalization. The model performs very well on the training set but not so good on the test set. 

![overfitting](https://hackernoon.com/hn-images/1*xWfbNW3arf39wxk4ZkI2Mw.png)

As we can see on training further a point the training error decreases and testing error increases.

A hypothesis h1 is said to overfit iff there exists another hypothesis h where h gives more error than h1 on training data and less error than h1 on the test data

## 12_ Bias & variance

Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. Model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to high error on training and test data.

Variance is the variability of model prediction for a given data point or a value which tells us spread of our data. Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before. As a result, such models perform very well on training data but has high error rates on test data.


Basically High variance causes overfitting and high bias causes underfitting. We want our model to have low bias and low variance to perform perfectly. We need to avoid a model with higher variance and high bias

![bias&variance](https://community.alteryx.com/t5/image/serverpage/image-id/52874iE986B6E19F3248CF?v=1.0)

We can see that for Low bias and Low Variance our model predicts all the data points correctly. Again in the last image having high bias and high variance the model predicts no data point correctly.

![B&v2](https://adolfoeliazat.com/wp-content/uploads/2020/07/Bias-Variance-tradeoff-in-Machine-Learning.png)

We can see from the graph that rge Error increases when the complex is either too complex or the model is too simple. The bias increases with simpler model and Variance increases with complex models.

This is one of the most important tradeoffs in machine learning



## 13_ Tree and classification

## 14_ Classification rate

## 15_ Decision tree


## 16_ Boosting

## 17_ Naïves Bayes classifiers

## 18_ K-Nearest neighbor

## 19_ Logistic regression

## 20_ Ranking

## 21_ Linear regression

## 22_ Perceptron

The perceptron has been the first model described in the 50ies.

This is a __binary classifier__, ie it can't separate more than 2 groups, and thoses groups have to be __linearly separable__.

The perceptron __works like a biological neuron__. It calculate an activation value, and if this value if positive, it returns 1, 0 otherwise.

## 23_ Hierarchical clustering

## 24_ K-means clustering

## 25_ Neural networks

## 26_ Sentiment analysis

## 27_ Collaborative filtering

## 28_ Tagging

## 29_ Support Vector Machine

## Reinforcement Learning
