# IMDb Moviw Review Sentiment Analysis - Naive Bayes Model

## Table of Contents
* [Naive Bayes Overview](#naive-bayes-overview)
* [Problem Statement](#problem-statement)
* [Technologies Used](#technologies-used)
* [Approach for Modeling](#approach-for-modeling)
* [Classification Outcome](#classification-outcome)
* [Conclusion](#conclusion)
* [Acknowledgements](#acknowledgements)

## Naive Bayes Overview

**Naive Bayes** is a classification technique based on **Bayes' Theorem**, which is used to predict the probability that a given instance belongs to a particular class. The Naive Bayes classifier assumes that the features (or attributes) are **conditionally independent**, given the class label, hence the name "naive". Despite this strong assumption of independence, it performs remarkably well in practice for many tasks, especially in text classification.

### Bayes' Theorem

At the core of Naive Bayes is **Bayes' Theorem**, which describes the probability of an event, based on prior knowledge of conditions that might be related to the event. Bayes' Theorem is given by:

$$
P(C|X) = \frac{P(X|C)P(C)}{P(X)}
$$

Where:
- $P(C|X)$ is the posterior probability of class $C$ given the feature vector $X$.
- $P(X|C)$ is the likelihood of the feature vector $X$ given class $C$.
- $P(C)$ is the prior probability of class $C$.
- $P(X)$ is the evidence or total probability of the feature vector $X$.

### Naive Bayes Classification

The "naive" assumption refers to the fact that the algorithm assumes that all features are **independent** of each other given the class label. This simplifies the calculation of the likelihood:

$$
P(X|C) = P(x_1|C) \times P(x_2|C) \times \cdots \times P(x_n|C)
$$

Where $X = (x_1, x_2, ..., x_n)$ is the feature vector and $x_i$ are the individual features. This strong assumption of independence, although not always true, leads to efficient computations.

To classify an instance, Naive Bayes computes the posterior probability for each class and selects the class with the highest posterior probability:

$$
\hat{C} = \underset{C}{\text{argmax}} \ P(C|X)
$$


**Naive Bayes (NB)** has been utilized in this case study in a step-by-step manner to understand, analyse, transform and model the data provided for the analysis. The approach described here represent the practical process utilised in industry to predict categorical target parameters for business.


## Problem Statement

This was one of the first widely-available **sentiment analysis** datasets compiled by Pang and Lee's. The data was first collected in 2002, however, the text is similar to movies reviews we find on IMDB today.

The dataset is in a CSV format. It has two categories: Pos (reviews that express a positive or favourable sentiment) and Neg (reviews that express a negative or unfavourable sentiment). For this exercise, we assume that all reviews are either positive or negative; there are no neutral reviews.

We need to build a **Multinomial Naive Bayes classification model** in Python for predicting the sentiment inherent to the reviews.

## Technologies Used

Python Jupyter Notebook with Numpy, Pandas, Matplotlib and Seaborn libraries are used to prepare, analyse and visualise data. The following model specific libraries have been used in the case study:

- scikit-learn


## Approach for Modeling

The following steps are followed to build the Naive Bayes model for IMDb sentiment analysis:

1. Import & Preprocess Data
2. Build Bernoulli's Model
3. Evaluate Bernoulli's Model
4. Build Multinomial Model
5. Evaluate Multinomial Model
6. Conclusion

Some distinguishing processes in this approach include,

- Vectorization of review texts after removing the stop words using `CountVectorizer` method

- Removal of words from the vector matrix which are rarely used (< 3%) in texts as well as extremely common ones which have appeared in 80% of the comments

- Evaluate model performance using the classification metrics like `sensitivity, specificity, accuracy` etc.

- Create `ROC Curve` to assess if `area under curve` indicates a descent model performance. 

- Perform prediction using `Test` data set. Evaluate if the `Accuracy` score is still about the same with good sensitivity, specificity, precision scores on `test` data set also

- Finally explain the model in business terms for easy comprehension of Business Users

## Classification Outcome

Multinomial Naive Bayes is slightly improved model over the Bernoulli's model. The model performances found are as follows:

* The model correctly predicts review sentiment with an **accuracy** of 83%
* Out of all positive reviews, it identifies 79.5% correctly (recall)
* Out of all predicted positive reviews, 85% are truly positive (precision)

The model works by identifying specific words that tend to show up in positive or negative reviews. For instance, if a review contains words like "like", "good", or "story", the model will predict a positive sentiment. If it finds words like "bad", "plot", or "waste", it will predict a negative sentiment.


## Conclusion

Naive Bayes is a simple yet powerful classification technique, especially useful for text classification and categorical data. It leverages the simplicity of Bayes' Theorem and the assumption of feature independence to perform classification in a fast and effective manner, even with limited computational resources.


## Acknowledgements

This case study has been developed as part of Post Graduate Diploma Program on Machine Learning and AI, offered jointly by Indian Institute of Information Technology, Bangalore (IIIT-B) and upGrad.