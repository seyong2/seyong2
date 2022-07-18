---
layout: post
title: Lasso Regression 
subtitle: Predict Weight of Fish Species using Multiple Variables
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, lasso regression]
comments: true
---

In today's article, we will look at lasso regression as an extension of ridge regression. They are very similar in that they aim to reduce variance at the expense of bias by penalizing a model with a penality term. The difference between them is that lasso regression takes the sum of absolute values of coefficients instead of the sum of squared coefficients. In other words, lasso regression minimizes $\sum_{i=1}^{N}(y_{i}-\hat{y}_{i})^{2}+\lambda\times\sum_{j=1}^{P}\hat{\beta}_{j}$ where $N$ is the data size, $\lambda$ is the regularization parameter and $P$ is the number of slope coefficients. 

Let's have a look at how this differnece in the penalty term makes lasso regression work differently from ridge regression. The comparison is done by means of mean squared error (MSE) from estimating the relationship between weight of bream fish and five independent variables. We start by loading necesssary libraries and the data. As always, we use the data about common fish species from [Kaggle](https://www.kaggle.com/datasets/aungpyaeap/fish-market?resource=download). 

```
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split

df = pd.read_csv('fish.csv')
df_bream = df.loc[df['Species']=='Bream', :]
df_bream.head()
```

![df_bream_head](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_multivariate_regression/df_bream_head.png?raw=true)

We define the features ($X$) and the dependent variable ($y$) and split them into training and test data. The training data contains two-thirds of the total observations, and the rest of the data is used to determine the amount of variance produced by various models.

```
X = df_bream.iloc[:, 2:]
y = df_bream['Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape)
```

We fit three different models to the data; ridge, lasso and multivariate regression models. To determine the optimal $\lambda$ value for the ridge and lasso regressions, we use 10-fold cross-validation for each value of the argument, 'alphas'. The values that produce the smallest MSE are 1.0 and 0.4, respectively, for the ridge and lasso regression models.

```
reg_ridge = RidgeCV(alphas=np.arange(0, 10, 0.1), scoring='neg_mean_squared_error', cv=10).fit(X_train, y_train)
reg_lasso = LassoCV(alphas=np.arange(0.1, 10, 0.1) , cv=10, max_iter=10**4).fit(X_train, y_train)
reg_no_penalty = LinearRegression().fit(X_train, y_train)

reg_ridge.alpha_
reg_lasso.alpha_
```

Now that we have three models fitted to the data, we examin their estimates of the coefficient parameters and MSE on the test data. As you can see the estimates, with lasso regression, are not reduced to zero, which means that all the predictors are relevant to the outcome variable, $Weight$. Also, their absolute values are greater than the values estimated by lasso regression, but smaller than the model without a penalty term. 

```
beta_hat_ridge = pd.DataFrame([reg_ridge.intercept_]+list(reg_ridge.coef_), index=['Intercept']+list(X.columns), columns=['beta_hat_ridge']).T
beta_hat_lasso = pd.DataFrame([reg_lasso.intercept_]+list(reg_lasso.coef_), index=['Intercept']+list(X.columns), columns=['beta_hat_lasso']).T
beta_hat_no_penalty = pd.DataFrame([reg_no_penalty.intercept_]+list(reg_no_penalty.coef_), index=['Intercept']+list(X.columns), columns=['beta_hat_no_penalty']).T
pd.concat([beta_hat_ridge, beta_hat_lasso, beta_hat_no_penalty])
```
![beta_hat_comparison](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_lasso_regression/beta_hat_comparison.png?raw=true)

Finally, we compare the MSE of the models. It can be seen from the table below that the ridge regression model performed best on the test data follwed by the lasso and multivariate regression models. This was an expected result because the coefficients estimated by the ridge regression model had the smallest absolute values. If the model had contained many useless variables, lasso regression could have removed them from the model. And this would have led the lasso regression to have a better performance than the ridge model.

```
def MSE(y, y_hat):
    return ((y-y_hat)**2).mean()

y_hat_ridge = reg_ridge.predict(X_test)
MSE_ridge = MSE(y_test, y_hat_ridge)

y_hat_lasso = reg_lasso.predict(X_test)
MSE_lasso = MSE(y_test, y_hat_lasso)

y_hat_no_penalty = reg_no_penalty.predict(X_test)
MSE_no_penalty = MSE(y_test, y_hat_no_penalty)

pd.DataFrame.from_dict({"MSE_ridge": [MSE_ridge], "MSE_lasso": [MSE_lasso], "MSE_no_penalty": [MSE_no_penalty]})
```
![MSE_comparison](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_lasso_regression/mse_comparison.png?raw=true)
