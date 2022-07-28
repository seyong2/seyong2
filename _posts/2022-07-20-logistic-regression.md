---
layout: post
title: Logistic Regression 
subtitle: Predict Whether or Not a User Clicked on Ad
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, logistic regression]
comments: true
---

Today's post is about logistic regression that predicts a categorical dependent variable given a set of independent variables. We will go through important concepts of logistic regression with data about advertisement downloaded from [Kaggle](https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad?resource=download). The goal here is to predict well whether a user clicked on an advertisement or not using logistic regression. Let's start by importing the data and libraries needed throughout the post.  

```
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('fish.csv')
df_bream = df.loc[df['Species']=='Bream', :]
df_bream.head()
print(df_bream.shape)
```

![df_bream_head](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_multivariate_regression/df_bream_head.png?raw=true)

We see that the data has 1,000 rows and 10 columns. Each row contains 10 different information of a user and the ten columns describe the following:

- **Daily Time Spent on Site**: Consumer time on site in minutes
- **Age**: Customer age in years
- **Area Income**: Avergae Income of geographical area of consumer
- **Daily Internet Usage**: Average minutes a day consumer is on the internet
- **Ad Topic Line**: Headline of the advertisement
- **City**: City of consumer
- **Male**: Whether or not consumer was male
- **Country**: Country of consumer
- **Timestamp**: Time at which consumer clicked on Ad or closed window
- **Clicked on Ad**: 0 or 1 indicated clicking on Ad

The target variable here is **Clicked on Ad** which is a binary variable. As an easy illustration, we are going to estimate the relationship between **Clicked on Ad** and **Daily Internet Usage**, a continuous measurement. The plot below shows how the values for the two variables are scattered. We see that the relationship cannot be estimated using linear regression. Instead, a *S*-shaped curve seems more appropriate in this case. That is what logistic regression exactly does by fitting **logistic function**. The function takes a value between 0 and 1, representing the probability that a consumer clicked on the ad given his/her Internet usage on a daily basis. Then, if the probability exceeds a certain threshold value of 50 percent, for example, we classify that the consumer clicked on the ad.

![scatter_plot](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_logistic_regression/scatterplot.png?raw=true)

Often we transform probability ($p$) to log odds ($log(\frac{p}{1-p})$) such that the *S*-shaped curve becomes a straight line which ranges between $+\infty$ and $-\infty$. This is illustrated by the following figure; while the probabilities are bounded between 0 and 1, the log odds do not have limits. Thus, we are not able to model probabilities just like in linear regression and this is the reason why we need the transformation. To find the best fitting line, we no longer use least squares method that minimizes sum of squared residuals (SSR) because the log odds of the outcomes (0 or 1) is infinite from which we cannot compute residuals. Instead, we use another method, which is **maximum likelihood**. We first obtain log odds of the data by projecting them onto a candidate line. Then, we transform the log odds to probabilities (or likelihoods) using $p=\frac{e^{log(odds)}}{1+e^{log(odds)}}$. Next, we multiply the likelihoods to get the overall likelihood, which is then transformed into the log of the likelihood for a easier computation. This can also be calculated by adding the logs of the individual likelihoods. The best line is, therefore, the one that has the largest log-likelihood value.

![prob_log_odds](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_logistic_regression/prob_log_odds.png?raw=true)

The resulting coefficient estimates of the best fitting line in terms of log odds are presented in the figure below. The intercept which approximately equals 8.73, is the log odds of clicking on the ad when **Daily Internet Usage** is zero. In other words, a consumer who has zero daily internet usage has 99.9 percent probability of clicking the ad, which is very high. On the other hand, the slope coefficient estimate indicates that a minute increase in daily internet usage is associated with a decrease of about 0.05 in the log of odds of clicking on the ad. Equally, an increase of 1 minute in daily Internet usage is associated with an increase of about 95 percent in the odds of clicking on the ad.

![beta_hat_logistic](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_logistic_regression/beta_hat_logistic.png?raw=true)

However, even if we know it is the line that fits best to the data, how do we know whether it is useful or not? In linear regression, this was measured by calculating $R^2$ and its corresponding $p$-value for the relationship between dependent and independent variables. In case of logistic regression, there are many ways to calculate $R^2$. Here, we use **McFadden's Pseudo R^2$**, which is quite similar to the way we compute $R^2$ for linear models. We saw that we had to transform probabilities into log odds values to estimate the model, which led the residuals to be infinite. Thus, we project the data onto the best fitting line and translate the log odds back to probabilities. Then, we compute the log-likelihood of the data, which we denote $LL(fit)$. We also need to compute $LL(mean)$, which is the log-likelihood of the data without considering **Daily Internet Usage**. This is based on the overall probability of clicking the ad. Consequently, like in linear models, $R^2$ is computed as $\frac{LL(mean)-LL(fit)}{LL(mean)-LL(saturated}$ where $LL(saturated)$ equals 0 for logistic regression. The range of $R^2$ is also 0 (worst fit) and 1 (best fit).

Regarding the $p$-value, we first need to compute $2((LL(saturated)-LL(fit)-(LL(saturated)-LL(mean))$, which is a Chi-squared value with degrees of freedom equal to the difference in the number of parameters in the two models. In this case, the degrees of freedom is one because we only have one extra term, **Daily Internet Usage**. As always, the $p$-value is the proability that we get at least as extreme as the statistic we have in this example. Since the $p$-value is small, we conclude that the relationship between **Clicked on Ad** and **Daily Internet Usage** is not due to some random chance.
