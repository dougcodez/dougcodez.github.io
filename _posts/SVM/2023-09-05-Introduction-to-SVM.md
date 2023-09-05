---
tag: $SVM$
title: Introduction to Support Vector Machines (SVM)
---

### Overview 
Support Vector Machines are a type of supervised learning algorithm that is used for classification or regression tasks. The use of
SVMs are to find a hyperplane that maximally separates the different classes in the given training data. We do this by finding 
the hyperplane with the largest margin, which is definesd as the given distance betweem the hyperplane and the closest data points
from each class. Once we have an idea where the hyperplane lies, new data can be classified by determining which side of
the hyperplane the data belongs to. SVMs are practically useful when the data contains many features, and/or when there is a 
clear margin of separation in the data. 

*Reminder do not get confused between SVM and logistic regression. Both the algorithms tryto find the best fitting hyperplane,
but logistic regression is a probabilistic approach whereas SVM is based on statistical approaches. 

### Types of SVM

**Linear SVM**

Linear SVM is used for linearly separable data, which means if a dataset can be classified into two classes by using a single straight
line, then such data is termed as linearly separable data, and classifier is used called as Linear SVM classifier.
  
![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/2886106f-b942-4389-8c42-791b7e821d0e)

Non-Linear SVM is used for non-linearly separated data, which means if a dataset cannot be classified by using a straight line, then
such data is termed as non-linear data and classifier used is called as Non-linear SVM classifier

![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/e64fa0b4-3bf9-4e22-aa27-77ff616f1a0e)

### Optimal Hyperplane
Imagine we have a dataset that has two tags (green and blue) and the dataset has two features x1 and x2. Our goal is to have a
classifier that can classify the pair (x1,x2) of coordinates in either green or blue. The SVM algorithm helps to find the best line
or decision boundary. SVM algorithms finds the closes point of the lines from both the classes. These points are called support
vectors. The given distance between the vectors and the hyperplane is called as margin, and the goal of SVM is to maximize this
margin. The hyperplane with maximum margine is called the optimal hyperplane.

![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/7c0a0ba9-b737-49e9-9998-31ba15f67ad7)

### Final
That pretty much sums up the simplicity behind Support Vector Machines. In later posts we'll be going over the different kernel types
of the SVM when using the sklearn package. 
