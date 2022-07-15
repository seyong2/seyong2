---
layout: post
title: Ridge Regression 
subtitle: Predict Weight of Fish Species using Multiple Variables
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, simple linear regression]
comments: true
---

In the previous post, we modeled the relationship between weight of bream fish and multiple variables using least squares method. We also compared $R^2$ of the multivariate regression model with that of a simple linear regression that considers only height. The $F$ test lead us to conclude that adding five extra variables to the model does not help much when it comes to predicting the weight. Thus, in today's post, the relationship is modeled via a ridge regressio that will allow us to see which variables contribute more to the model. 

Ridge regression is used a lot when there are no enough data for training a model. If the data size is small, it is highly likely that the resulting model has high variance even if it has small bias. In other words, the model tends to overfit to the training data, meaning that the model has a very good fit on the training data but at the same time it produes poor predictions on new data. This problem can be prevented by means of ridge regression that tries to reduce variance by fining a model that fits worse the training data. Thus, even if we gain a small amount of bias, we are able to obtain a significant drop in variance at the same time.

Recall the multivariate regression model from the previous post.
$\hat{Weight}=\hat{\beta}_0 + \hat{\beta}_1Length1 + \hat{\beta}_2Length2 + \hat{\beta}_3Length3 + \hat{\beta}_4Height + \hat{\beta}_5Width$
Instead of minimizing the sum of squared residuals (SSR) that does least squares method, ridge regression estimates the parameters minimizing not only the SSR but also $\lambda\sum_{i=1}^{5}\hat{\beta}_i^2$ where $\lambda$ is regularization penalty that determines the amount of penalty given to the least squares method. $\lambda$ can take value between 0 and positive infinity and the larger its value is, the more severe the penalty is. To obtain the optimal value for $\lambda$, we use 10-fold cross validation and find the value that produces the lowest variance. 

The parameter estimates using ridge regression are in general smaller than those using the least squares method. This indicates that predictions made by ridge regression model are usually less sensitive to changes in predictors than the least squares model.
