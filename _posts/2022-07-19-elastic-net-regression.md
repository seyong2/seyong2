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

What if we have a model with millions of variables that we don't know in advance whether or not they will be useful? Which one should we choose to estimate them, Ridge or Lasso? To avoid having to make this choice, we can use Elastic-Net regression. Starting with Least Squares, Elastic-Net regression combines Ridge and Lasso regression penalties, that is to say it minimizes $\sum_{i=1}^{N}(y_{i}-\hat{y}_{i})^2+\lambda_{L}\sum_{j=1}^{P}|\hat{\beta}_{j}|+\lambda_{R}\sum_{j=1}^{P}\hat{\beta}_{j}^{2}$ where $N$ is the size of data, $y_i$ is the $i$-th observed value of dependent variable, $\hat{y}_i$ is the corresponding predicted value, $P$ is the number of variables, $\hat{\beta}_{j}$ is the $j$-th predictor, $\lambda_{L}$ and $\lambda_{R}$ are the regularization constants for Lasso and Ridge regression, respectively. 

Elastic-Net regression is especially useful when the predictors are correlated. Through Lasso penalty term, one of the correlated variables is chosen and the others are removed. Ridge penalty term also makes coefficients for the correlated predictors small. The combination of the two, therefore, leads Elastic-Net regression to be better at handling correlated variables.

Now we are going to see the example of fish market and assess how Elastic-Net regression works. We start by importing necessary libraries and the data. As always, we want to estimate the relationship between weight of bream fish ($Weight$) and five characteristics; vertical length in cm ($Length1$), diagonal length in cm ($Length2$), cross length in cm ($Length3$), height in cm ($Height$), and diagonal width in cm ($Width$).

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

We define $X$ as a matrix of the predictors and $y$ as a vector of the dependent variable. We then divide the test data into training data and test data, where test data accounts for 1/3 of the total observations.

```
X = df_bream.iloc[:, 2:]
y = df_bream['Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

Then, we fit an Elastic-Net regression model to the data. The argument "l1_ratio" is a number between 0 and 1 representing the ratio between the Lasso and Ridge penalties. Then if the value is 1, only the Lasso penalty is considered, otherwise 0. We test different values using 10-fold cross-validation to obtain the optimal ratio value.

```
reg_ela = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=10, max_iter=10**4).fit(X_train, y_train)

reg_ela.alpha_
reg_ela.l1_ratio_
```

The optimal value for "l1_ratio" is 0.9, meaning that the proportion of the lasso penalty term accounts for 90 percent. The resulting "alpha_" value is approximately 0.83, which corresponds to $\lambda_{L}$. Then, $\lambda_{R}$ is $1-0.83=0.17$. 

We now compare the performance of the Elastic-Net regression and three other models, Ridge, Lasso and multivariate models, by menas of the estimated parameter coefficients and MSE.

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

The table above shows the coefficient estimates by the four models. Using Elastic-Net regression, we can see that the estimates for $Length1$ and $Length3$ are zero. This is because the predictor variables are highly correlated as we have seen before and those variables do not contribute to the model. 

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

Finally, looking at the MSE of the model, Elastic-Net regression is the regression that produces the smallest MSE in the test data. This can be attributed to the fact that Elastic-Net regression combines the advantages of Ridge and Lasso regression.
