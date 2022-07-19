---
layout: post
title: Elastic Net Regression 
subtitle: Predict Weight of Fish Species using Multiple Variables
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, elastic net regression]
comments: true
---

In the last two posts, we have seen two regularization techniques; Ridge and Lasso. Depending on the situation, we can choose one of the two to estimate a model. When the model contains many variables that do not help predict the dependent variable, Lasso regression works best because it removes those useless variables by making their coefficients zero. On the other hand, we will pick Ridge regression if most of the variables are useful in the model. It helps to reduce the variance of the model while shrinking the parameters close, but not zero.

What if we have a model with millions of variables that we don't know in advance whether or not they will be useful? Which one should we choose to estimate them, Ridge or Lasso? To avoid having to make this choice, we can use Elastic-Net regression. Starting with Least Squares, Elastic-Net regression combines Ridge and Lasso regression penalties, that is to say it minimizes the follwing expression $\sum_{i=1}^{N}(y_{i}-\hat{y}_{i})^2+\lambda_{L}\sum_{j=1}^{P}|\hat{\beta}_{j}|+\lambda_{R}\sum_{j=1}^{P}\hat{\beta}_{j}^{2}$ where $N$ is the size of data, $y_i$ is the $i$-th observed value of dependent variable, $\hat{y}_i$ is the corresponding predicted value, $P$ is the number of variables, $\hat{\beta}_{j}$ is the $j$-th predictor, $\lambda_{L}$ and $\lambda_{R}$ are the regularization constants for Lasso and Ridge regression, respectively. 

Elastic-Net regression is especially useful when the predictors are correlated. Through Lasso penalty term, one of the correlated variables is chosen and the others are removed. Ridge penalty term also makes coefficients for the correlated predictors small. 
