---
layout: post
title: Ridge Regression 
subtitle: Predict Weight of Fish Species using Multiple Variables
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, simple linear regression]
comments: true
---

In the previous post, we modeled the relationship between weight of bream fish and several characteristics using least squares method. However, we also saw that some variables were highly correlated with each other (see figure below). This multicollinearity can cause problems regarding fitting the model and interpreting the results. For example, estimate of a regression coefficient using the least squares method is the interpreted as the average change in the dependent variable due to unit change in that predictor when the other independent variables remain constant. But, if the predictors are correlated, shifts in one variable also lead to changes in the others. Then, it becomes complicated for a model to estimate the relationship between the outcome variable and each independent variable because depending on the variables included in the model, the coefficient estimates will alter a lot. Therefore, in today's post, we will have a look at ridge regression model which is one of the ways to overcome this problem of multicollinearity. If you need more information about multicollinearity, you can found them [here](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/). 

![corr_mat](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_multivariate_regression/corr_mat.png?raw=true)

We star by loading necessary libraries and data as usual.

```
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

df = pd.read_csv('fish.csv')
df_bream = df.loc[df['Species']=='Bream', :]
df_bream.head()
```

![df_bream_head](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_multivariate_regression/df_bream_head.png?raw=true)

As in the previous post, we assign $Weight$ to the dependent variable, $y$ and the remaining variables except $Species$ to $X$. And in order to see how the ridge regression model works on the data not used for training, we split the data into training and test set. The test data will contain 33% of the total observations. We also set the seed for the random generator to 1 for reproducibility. Then, the data for training and testing have 23 and 12 observations, respectively. 

```
X = df_bream.iloc[:, 2:]
y = df_bream['Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

Ridge regression is typically used when there are no enough data for training a model. If the data size is small, it is highly likely that the resulting model has high variance even if it has small bias. In other words, the model tends to overfit to the training data, meaning that the model has a very good fit on the training data but at the same time it produes poor predictions on new data. This problem can be prevented by means of ridge regression that tries to reduce variance by fining a model that fits worse the training data. Thus, even if we gain a small amount of bias, we are able to obtain a significant drop in variance at the same time.

Recall the multivariate regression model from the previous post.
$\hat{Weight}=\hat{\beta}_0 + \hat{\beta}_1Length1 + \hat{\beta}_2Length2 + \hat{\beta}_3Length3 + \hat{\beta}_4Height + \hat{\beta}_5Width$
Instead of minimizing the sum of squared residuals (SSR) that does least squares method, ridge regression estimates the parameters minimizing not only the SSR but also $\lambda\sum_{i=1}^{5}\hat{\beta}_i^2$ where $\lambda$ is regularization penalty that determines the amount of penalty given to the least squares method. $\lambda$ can take value between 0 and positive infinity and the larger its value is, the more severe the penalty is. To obtain the optimal value for $\lambda$, we use 10-fold cross validation and find the value that produces the lowest variance. 

The parameter estimates using ridge regression are in general smaller than those using the least squares method. This indicates that predictions made by ridge regression model are usually less sensitive to changes in predictors than the least squares model. Then, now we fit a ridge regression model to the data.

```
reg_ridge = RidgeCV(alphas=np.arange(0, 10, 0.1), scoring='neg_mean_squared_error', cv=10).fit(X_train, y_train)
reg_ridge.alpha_
```

The argument 'alphas' in function *RidgeCV* is the list of possible values for the regularization parameter, $\lambda$. We use 10-fold cross validation to obtain the one that produces the smallest average mean squared error (MSE) as specified in argument 'scoring'. After the training, the optimal value of $lambda$ is found to be 1.0, meaning that imposing penalty on the coefficients imporves the model performance. To verify it, we also fit a multivariate regression model to the training data and compare the results.

```
reg_no_ridge = LinearRegression().fit(X_train, y_train)

beta_hat_ridge = pd.DataFrame([reg_ridge.intercept_]+list(reg_ridge.coef_), index=['Intercept']+list(X.columns), columns=['beta_hat_ridge']).T
beta_hat_no_ridge = pd.DataFrame([reg_no_ridge.intercept_]+list(reg_no_ridge.coef_), index=['Intercept']+list(X.columns), columns=['beta_hat_no_ridge']).T
pd.concat([beta_hat_ridge, beta_hat_no_ridge])

def SSR(y, y_hat):
    return ((y-y_hat)**2).mean()

y_hat_ridge = reg_ridge.predict(X_test)
SSR(y_test, y_hat_ridge)
y_hat_no_ridge = reg_no_ridge.predict(X_test)
SSR(y_test, y_hat_no_ridge)
```

![beta_hat_comparison]()




