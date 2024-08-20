---
layout: post
title: Elastic Net Regression 
subtitle: Predict Weight of Fish Species using Multiple Variables
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, elastic net regression]
comments: true
---

In the last two posts, we have seen two regularization techniques; Ridge and Lasso. Depending on the situation, we can choose one of the two to estimate a model. When the model contains many variables that do not help predict the dependent variable, lasso regression works best because it removes those useless variables by making their coefficients zero. On the other hand, we would pick ridge regression if most of the variables are useful in the model. It helps to reduce the variance of the model while shrinking the parameters close, but not zero.

What if we have a model with millions of variables that we don't know in advance whether or not they will be useful? Which one should we choose to estimate them, ridge or lasso? To avoid having to make this choice, we can use elastic-net regression. Starting with least squares, elastic-net regression combines ridge and lasso regression penalties, that is to say, it minimizes 

$min_{\beta}\sum_{i=1}^{n}(y_{i}-X_{i}\beta)^2+\lambda_{L}\sum_{j=1}^{p}|\beta_{j}|+\lambda_{R}\sum_{j=1}^{p}\beta_{j}^{2}$ 

where $n$ is the size of data, $y_i$ is the $i$-th observed value of dependent variable, $X_{i}\beta$ is the corresponding predicted value, $p$ is the number of variables, $\beta_{j}$ is the $j$-th predictor's coefficient, $\lambda_{L}$ and $\lambda_{R}$ are the regularization constants for lasso and ridge regression, respectively. 

Elastic-net regression is especially useful when the predictors are correlated. Through Lasso penalty term, one of the correlated variables is chosen (although arbitrarily) and the others are removed. The ridge penalty term also makes coefficients for the correlated predictors small. The combination of the two, therefore, leads elastic-net regression to be better at handling multicollinearity.

Now we are going to see the example of fish market and assess how elastic-net regression works. We start by importing the necessary libraries and the data. As always, we want to estimate the relationship between the weight of bream fish ($Weight$) and five characteristics; vertical length in cm ($Length1$), diagonal length in cm ($Length2$), cross length in cm ($Length3$), height in cm ($Height$), and diagonal width in cm ($Width$).

```
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split

df = pd.read_csv('fish.csv')
df_bream = df.loc[df['Species']=='Bream', :]
df_bream.head()
```

![df_bream_head](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_multivariate_regression/df_bream_head.png?raw=true)

We define $X$ as a matrix of the predictors and $y$ as a vector of the dependent variable. We then divide the test data into training data and test data, where test data accounts for one third of the total observations.

```
X = df_bream.iloc[:, 2:]
y = df_bream['Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

Then, we fit an elastic-net regression model to the data. The argument "l1_ratio" is a number between 0 and 1 representing the ratio between the lasso and ridge penalties. Then if the value is 1, only the Lasso penalty is considered, otherwise 0. We test different values using 10-fold cross-validation to obtain the optimal ratio value.

```
reg_ela = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=10, max_iter=10**4).fit(X_train, y_train)

reg_ela.alpha_
reg_ela.l1_ratio_
```

The optimal value for "l1_ratio" is 0.9, meaning that the proportion of the lasso penalty term accounts for 90 percent. Therefore, we have $\lambda_{L}$ and $\lambda_{R}$ equal to 0.9 and 0.1, respectively.

We now compare the performance of the elastic-net regression and three other models, ridge, lasso, and multivariate models, through the estimated parameter coefficients and MSE.

```
reg_ridge = RidgeCV(alphas=np.arange(0, 10, 0.1), scoring='neg_mean_squared_error', cv=10).fit(X_train, y_train)
beta_hat_ridge = pd.DataFrame([reg_ridge.intercept_]+list(reg_ridge.coef_), index=['Intercept']+list(X.columns), columns=['beta_hat_ridge']).T

reg_lasso = LassoCV(alphas=np.arange(0.1, 10, 0.1) , cv=10, max_iter=10**4).fit(X_train, y_train)
beta_hat_lasso = pd.DataFrame([reg_lasso.intercept_]+list(reg_lasso.coef_), index=['Intercept']+list(X.columns), columns=['beta_hat_lasso']).T

beta_hat_ela = pd.DataFrame([reg_ela.intercept_]+list(reg_ela.coef_), index=['Intercept']+list(X.columns), columns=['beta_hat_ela']).T
beta_hat_ela

reg_no_penalty = LinearRegression().fit(X_train, y_train)
beta_hat_no_penalty = pd.DataFrame([reg_no_penalty.intercept_]+list(reg_no_penalty.coef_), index=['Intercept']+list(X.columns), columns=['beta_hat_no_penalty']).T


pd.concat([beta_hat_ridge, beta_hat_lasso, beta_hat_ela, beta_hat_no_penalty])
```

![beta_hat_comparison](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_elastic_net_regression/beta_hat_comparison.png?raw=true)

The table above shows the coefficient estimates by the four models. Using the elastic-net regression, we can see that the estimates for $Length1$ and $Length3$ are zero. This is because the predictor variables are highly correlated as we have seen before and those variables do not contribute to the model. 

```
def MSE(y, y_hat):
    return ((y-y_hat)**2).mean()

y_hat_ridge = reg_ridge.predict(X_test)
MSE_ridge = MSE(y_test, y_hat_ridge)

y_hat_lasso = reg_lasso.predict(X_test)
MSE_lasso = MSE(y_test, y_hat_lasso)

y_hat_ela = reg_ela.predict(X_test)
MSE_ela = MSE(y_test, y_hat_ela)

y_hat_no_penalty = reg_no_penalty.predict(X_test)
MSE_no_penalty = MSE(y_test, y_hat_no_penalty)

pd.DataFrame.from_dict({"MSE_ridge": [MSE_ridge], "MSE_lasso": [MSE_lasso], "MSE_ela": [MSE_ela], "MSE_no_penalty": [MSE_no_penalty]})
```

![mse_comparison](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_elastic_net_regression/mse_comparison.png?raw=true)

Finally, looking at the MSE of the model, the elastic-net regression model is the one that produces the smallest MSE in the test data. This can be attributed to the fact that the elastic-net regression combines the advantages of ridge and lasso regression.
