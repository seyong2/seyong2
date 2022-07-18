---
layout: post
title: Lasso Regression 
subtitle: Predict Weight of Fish Species using Multiple Variables
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, lasso regression]
comments: true
---

In today's article, we will look at lasso regression as an extension of ridge regression. They are very similar in that they aim to reduce variance at the expense of bias by penalizing a model with a penality term. The difference between them is that lasso regression takes the sum of absolute values of coefficients instead of the sum of squared coefficients. In other words, lasso regression minimizes $\sum_{i=1}^{N}(y_i-\hat{y}_i)^2+\lambda\times\sum_{j=1}^P\hat{\beta}_j$ where $N$ is the data size, $\lambda$ is the regularization parameter and $P$ is the number of slope coefficients. 

Let's have a look at how this differnece in the penalty term makes lasso regression work differently from ridge regression. 

As you can see the estimated coefficient parameters, with lasso regression, some estimates were reduced to zero, which did not occur with ridge regression. Then, lasso regression can then remove unhelpful predictors from models with many useless variables, which leads to a better performance on new data than ridge regression. Therefore, on the contrary, if most variables in the model are useful, ridge regression will do better.
