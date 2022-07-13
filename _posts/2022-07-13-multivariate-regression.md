---
layout: post
title: Multivariate Regression 
subtitle: Predict Weight of Fish Species using Multiple Variables
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, simple linear regression]
comments: true
---

In this post, we want to estimate the weight of a species of fish called bream. This is done through a simple linear regression model using their height. A more detailed description of the data can be found on [Kaggle](https://www.kaggle.com/datasets/aungpyaeap/fish-market?resource=download).

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

We use a scatterplot to investigate the relationship between the weight and height of the fish.

```
sns.scatterplot(x=df.loc[:, 'Height'], y=df.loc[:, 'Weight'], hue=df.loc[:, 'Species'])
```
![scatter_fish](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_simple_linear_regression/scatter_fish.png?raw=true)

The figure above shows that there is a positive relationship between the height and weight of all fish species. Since we are only interested in a species called bream, we slice the data.

```
df_bream = df[df.loc[:, 'Species'] == 'Bream']
x = df_bream.loc[:, 'Height']
y = df_bream.loc[:, 'Weight']
sns.scatterplot(x=x, y=y)
```

![scatter_bream](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_simple_linear_regression/scatter_bream.png?raw=true)

It seems that we can add a line to the data specific to the bream. But, how can we draw the line that best describes the data? To get an idea, a horizontal line is drawn that cuts through the average weight and this is obviously the worst line one can have. 

```
scatter = sns.scatterplot(x=x, y=y)
scatter.axhline(y.mean(), color='r')
plt.show()
```

![scatter_bream_horizontal](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_simple_linear_regression/scatter_bream_horizontal.png?raw=true)


We can measure how well this horizontal line fits the data by calculating the total distance between the line and the data points. However, data points above the mean line have negative distances, which make the overall fit to appear better than it really is. Therefore, the residual sum of squares (SSR) is calculated by squaring and summing the distances.

```
def SSR(y, y_hat):
    err = 0
    for i in range(len(y)):
        err += (y[i] - y_hat[i])**2
    return err

SSR(y, [y.mean()]*len(y))
```

The resulting SSR for the horizontal line is around 1,488,078.97, which looks quite large. But if we rotate the line we can get the line with smaller SSR although if you rotate too much the SSR will grow again. As a result, we need to find a sweet spot in-between at which the SSR has no slope. This is what the least squares method does to estimate the optimal line. The following figure shows such a line and it best describes the data. The intercept of the optimal line is approximately -941.56, which means that a zero-height fish weighs that much. This does not make sense in practice so we have to be aware of extrapolation. The optimal slope is equal to about 102.70. That is, an increase in height by one unit increases the weight by 103 g.

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
The fitted line with the least squares estimates has an SSR of 103,699.21, so it fits much better than the horizontal line. How well does the fitted line work better than the mean line? This question can be answered by means of $R^2=\frac{Var(mean)-Var(line)}{Var(mean)}$. The metric takes a value between 0 and 1. 0 means that height does not help explain weight changes, and 1 means the opposite.

```
def R2(SSR_mean, SSR_line):
    return (SSR_mean - SSR_line)/SSR_mean

R2(SSR(y, [y.mean()]*len(y)), SSR(y, y_hat))
```

Here, the value of $R^2$ is 0.93, indicating that the relationship between height and weight accounts for nearly 93% of the total variance. But is this value statistically significant? To determine whether or not it is significant, we need to compute the $p$-value for the $F$-statistic defined as $\frac{SS(mean)-SS(fit)/(p_{fit}-p_{mean})}{SS(fit)/(n-p_{fit})}$ where where $n$ is the size of the data, $p_{fit}$ is the number of parameters in the fitted line and $p_{mean}$ is the number of parameters in the mean line. The numerator, then, is the variance of fish weight explained by the height and the denominator is the amount of variation that remains unexplained. So, really large values of the $F$ statistic indicate a good fit of the line. For the $p$-value, we use the $F$-distribution to calculate the probability of obtaining an $F$ statistic at least as extreme as the observed statistic.

```
def F_stat(SSR_mean, SSR_fit, n, p_fit, p_mean):
    return ((SSR_mean-SSR_fit)/(p_fit-p_mean)) / ((SSR_fit)/(n-p_fit))

F = F_stat(SSR(y, [y.mean()]*len(y)), SSR(y, y_hat), x.shape[0], 2, 1)

p_val = 1-f.cdf(F, 2-1, x.shape[0]-2)
p_val
```

In this example, the $p$-value is very close to 0, which is much less than the significance level of 0.05. Consequently, we conclude that $R^2$ is significant and that the height of the fish explains much of the variation in weight.
