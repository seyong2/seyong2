---
layout: post
title: Multivariate Regression 
subtitle: Predict Weight of Fish Species using Multiple Variables
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, multivariate regression]
comments: true
---

In this post, we want to estimate the weight of a species of fish called bream as done in the previous one. But, this time we use not only height but also four more characteristics of the fish. For more details about the data, please refer to [Kaggle](https://www.kaggle.com/datasets/aungpyaeap/fish-market?resource=download) or [my previous post](https://seyong2.github.io/2022-07-10-simple-linear-regression/).

As usual, we start by loading the necessary libraries and data.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import f

df = pd.read_csv('fish.csv')
df_bream = df.loc[df['Species']=='Bream', :]
df_bream.head()
print(df_bream.shape)
```
![df_bream_head](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_multivariate_regression/df_bream_head.png?raw=true)

The data include 7 traits for 35 bream fish. The description of the columns are as follows:

- *Species*: species name of fish
- *Weight*: weight of fish in g
- *Length1*: vertical length in cm
- *Length2*: diagonal length in cm
- *Length3*: cross length in cm
- *Height*: height in cm
- *Width*: diagonal width in cm

We are going to use all the numerical variables to train a multivariate regression model that predicts weight. Since there are in total 5 independent variables, we fit a hyperplane instead of a line that we had in case of a simple linear regression. Then, the regression function can be expressed as follows: $\hat{Weight}=\hat{\beta}_0+\hat{\beta}_1Length1+\hat{\beta}_2Length2+\hat{\beta}_3Length3+\hat{\beta}_4Height+\hat{\beta}_5Width$.

```
X_multiple = df_bream.iloc[:, 2:]
y = df_bream.iloc[:, 1]
X_multiple.head()

```
![X_multiple_head](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_multivariate_regression/X_multiple_head.png?raw=true)

The least squares method is used to estimate parameters $\beta$=($\beta_0$, $\beta_1$, $\beta_2$, $\beta_3$, $\beta_4$, $\beta_5$) by minimizing sum of squared residuals (SSR). If the variables are useless for predicting the weight of the fish, that is, if they do not make the SSR any smaller, the method will make their slope set to zero. This implies that adding extra parameters can never result in worse SSR.

```
reg_multiple = LinearRegression()
reg_multiple.fit(X_multiple, y)

pd.DataFrame([reg_multiple.intercept_]+list(reg_multiple.coef_), index=['Intercept']+list(X_multiple.columns), columns=['beta_hat']).T
```

![beta_hat](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_multivariate_regression/beta_hat.png?raw=true)

The parameter estimates are shown in the table above. Note that the least square estimate for $\beta_4$ is quite different from the one that we obtained using the simple linear regression (102.70). This is due to multicollinearity where the variables are correlated with each other, which can also be seen in the correlation matrix below. 

```
sns.heatmap(X_multiple.corr())
```
![corr_mat_tab](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_multivariate_regression/corr_mat_tab.png?raw=true)

![corr_mat](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_multivariate_regression/corr_mat.png?raw=true)

```
def SSR(y, y_hat):
    err = 0
    for i in range(len(y)):
        err += (y[i] - y_hat[i])**2
    return err

def R2(SSR_mean, SSR_line):
    return (SSR_mean - SSR_line)/SSR_mean
    
SSR_mean = SSR(y, [y.mean()]*len(y))
SSR_multiple = SSR(y, reg_multiple.predict(X_multiple))
R2(SSR_mean, SSR_multiple)
```

To assess how well the hyperplane fits the data compared to the mean of fish weight, we compute $R^2$. It gives us 0.9432570487861149, meaning that almost 94% of total variation in the weight is expained by the variables. However, it happens sometimes that even if some predidctors are worthless, there are small probabilities that they contribute to predicting outcome variable due to random chance. This in turn leads to better $R^2$, which is not supposed to be. Consequently, we need to adjust the $R^2$ by the number of the variables considered in the model and the new metric is defined as $R^2_{adj}=1-\frac{SSR_{fit}/(n-p_{fit})}{SSR_{mean}/(n-1)}$.

```
def R2_adj(SSR_mean, SSR_fit, n, p_fit):
    return 1-(SSR_fit/(n-p_fit-1))/(SSR_mean/(n-1))

R2_adj(SSR_mean, SSR_multiple, len(y), X_multiple.shape[1])
```

Here, the value of $R^2_{adj}$ is 0.9334737813354451 and its difference with the unadjusted one is small. Now we test whether this value is signficiant or not through a $F$ test.

```
def F_stat(SSR_mean, SSR_fit, n, p_fit, p_mean):
    return ((SSR_mean-SSR_fit)/(p_fit-p_mean)) / ((SSR_fit)/(n-p_fit))

F = F_stat(SSR_mean, SSR_multiple, len(y), X_multiple.shape[1]+1, 1)

p_fit = X_multiple.shape[1]+1
p_mean = 1
n = len(y)
p_val = 1-f.cdf(F,p_fit-p_mean, n-p_fit)
p_val
```

In this example, the $p$-value is very close to 0 and is much smaller than the significance level of 0.05. Therefore, we conclude that the five variables explain much of the variation in weight. Until now, we have compared the multivariate regression to the mean but, we can also do the same with a simple linear regression. A comparison between fits with and without the additional variables will tell us if it worh including them in the model.

```
X_simple = X_multiple['Height']
reg_simple = LinearRegression() 
reg_simple.fit(np.array(X_simple).reshape((-1,1)), y)

SSR_simple = SSR(y, reg_simple.predict(np.array(X_simple).reshape((-1,1))))

F = F_stat(SSR_simple, SSR_multiple, len(y), X_multiple.shape[1]+1, 2)

p_multiple = X_multiple.shape[1]+1
p_simple = 2
n = len(y)
p_val = 1-f.cdf(F, p_multiple-p_simple, n-p_multiple)
p_val
```
The resulting $p$-value is approximately 0.19, which leads us to a conclustion that we are good enough with the simple regression model.
