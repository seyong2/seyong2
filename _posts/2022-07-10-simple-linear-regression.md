---
layout: post
title: Simple Linear Regression 
subtitle: Predict Weight of Fish Species Using Height
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, simple linear regression]
comments: true
---

In this post, we want to estimate the weight of a species of fish called bream. This is done through a simple linear regression model using their height. A more detailed description of the data can be found on [Kaggle](https://www.kaggle.com/datasets/aungpyaeap/fish-market?resource=download).

# Import Libraries and Data

We start by loading the necessary libraries and data.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import f

df = pd.read_csv('fish.csv')
df.head()
```
![df_head](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_simple_linear_regression/df_head.png?raw=true)

The data include 7 traits for 159 fish in the market. The description of the columns are as follows:

- *Species*: species name of fish
- *Weight*: weight of fish in g
- *Length1*: vertical length in cm
- *Length2*: diagonal length in cm
- *Length3*: cross length in cm
- *Height*: height in cm
- *Width*: diagonal width in cm

We use a scatterplot to represent the relationship between the weight and height of the fish.
```
sns.scatterplot(x=df.loc[:, 'Height'], y=df.loc[:, 'Weight'], hue=df.loc[:, 'Species'])
```
![scatter_fish](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_simple_linear_regression/scatter_fish.png?raw=true)

The figure above shows that there is a positive relationship between the height and weight of all fish species. We are only interested in a species called Bream, so we slice the data.

```
df_bream = df[df.loc[:, 'Species'] == 'Bream']
x = df_bream.loc[:, 'Height']
y = df_bream.loc[:, 'Weight']
sns.scatterplot(x=x, y=y)
```
![scatter_bream](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_simple_linear_regression/scatter_bream.png?raw=true)

It seems that we can add a line to the data to see the trend. But, how can we draw the line that best describes the data? First, a horizontal line is drawn that cuts through the average weight. It is likely that this is the the worst line that one can have. However, we can get an idea about finding the optimal line.

```
scatter = sns.scatterplot(x=x, y=y)
scatter.axhline(y.mean(), color='r')
plt.show()
```

![scatter_bream_horizontal](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_simple_linear_regression/scatter_bream_horizontal.png?raw=true)


We can measure how well this horizontal line fits the data by calculating the total distance between the line and the data points. However, when the data point is above the line, the distance is negative, which makes the overall fit appear better than it really is. Thus, we compute sum of squared residuals (SSR) by squaring the distances and summing them up.

```
def SSR(y, y_hat):
    err = 0
    for i in range(len(y)):
        err += (y[i] - y_hat[i])**2
    return err

SSR(y, [y.mean()]*len(y))
```

The resulting SSR for the horizontal line is 1488078.9714285715. Then, by rotating the line, we can obtain a line (intercept and slope) whose SSR is the smallest. However, if we rotate too much, the fit gets worse again so we need to find the sweet spot in-between at which the function SSR has no slope. The figure below shows the line that fits best the data. The optimal line has an intercept of -941.559004487088, meaning that a fish whose height is zero weighs approximately -942g. This does not make sense in practice so we have to be aware of extrapolation. The slope is equal to 102.70472642, that is, a unit increase in the height leads to an increase of 103g in the weight.

```
reg = LinearRegression()
reg.fit(np.array(x).reshape((-1, 1)), y)
y_hat = reg.intercept_ + reg.coef_*x
sns.scatterplot(x=x, y=y)
sns.lineplot(x=x, y=y_hat, color='r')
plt.show()
```

![scatter_bream_best](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_simple_linear_regression/scatter_bream_best.png?raw=true)

```
SSR(y, y_hat)
```
We see that the SSR of the fitted line with the least squares estimates is 103699.20790298669 so it fits much better than the horizontal line. How much does better the fitted line does its job than the mean line? The difference can be quantified by means of $R^2=\frac{Var(mean)-Var(line)}{Var(mean)}$. The metric can take a value between 0 and 1, 0 meaning that height does not help explaining the variation in weight and 1, meaning that the line has the perfect fit. 

```
def R2(SSR_mean, SSR_line):
    return (SSR_mean - SSR_line)/SSR_mean

R2(SSR(y, [y.mean()]*len(y)), SSR(y, y_hat))
```

The $R^2$ value here is equal to 0.9303133705307088 and this indicates that the relationship between the height and weight accounts for almost 93% of the total variation. But, is this value statistically significant? To determine whether or not it is significant, we need to compute a $p$-value for $F$-statistic defined as $\frac{SS(mean)-SS(fit)/(p_{fit}-p_{mean})}{SS(fit)/(n-p_{fit})}$ where where $n$ is the size of the data, $p_{fit}$ is the number of parameters in the fit line and $p_{mean}$ is the number of parameters in the mean line. The numerator, then, is the variation in fish weight explained by height and the denominator is the variation left to be explained. Thus, a really large value of $F$ indicates that the fit of the line is good. For the $p$-value, we calculate the probability of obtaining $F$ statistics at least as extreme as the observed statistic using $F$-distribution.

```
def F_stat(SSR_mean, SSR_fit, n, p_fit, p_mean):
    return ((SSR_mean-SSR_fit)/(p_fit-p_mean)) / ((SSR_fit)/(n-p_fit))

F = F_stat(SSR(y, [y.mean()]*len(y)), SSR(y, y_hat), x.shape[0], 2, 1)

p_val = 1-f.cdf(F, 2-1, x.shape[0]-2)
p_val
```

The $p$-value is 1.1102230246251565e-16, which is much smaller than the significance level, 0.05. Therefore, we conclude that $R^2$ is significant and that the fish height explains much of the variation in weight.
