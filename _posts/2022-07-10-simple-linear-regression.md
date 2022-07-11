---
layout: post
title: Simple Linear Regression 
subtitle: Predict Weight of Fish Species Using Height
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [test]
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

df = pd.read_csv('fish.csv')
df.head()
```
![df_head](https://github.com/seyong2/seyong2.github.io/blob/master/_posts/figures/df_head.png)

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
![scatter_fish](https://github.com/seyong2/seyong2.github.io/blob/master/_posts/figures/scatter_fish.png)

The figure above shows that there is a positive relationship between the height and weight of all fish species. We are only interested in a species called Bream, so we slice the data.

```
df_bream = df[df.loc[:, 'Species'] == 'Bream']
x = df_bream.loc[:, 'Height']
y = df_bream.loc[:, 'Weight']
sns.scatterplot(x=x, y=y)
```
![scatter_bream](https://github.com/seyong2/seyong2.github.io/blob/master/_posts/figures/scatter_bream.png)

It seems that we can add a line to the data to see the trend. But, how can we draw the line that best describes the data? First, a horizontal line is drawn that cuts through the average weight. It is likely that this is the the worst line that one can have. However, we can get an idea about finding the optimal line.

```
scatter = sns.scatterplot(x=x, y=y)
scatter.axhline(y.mean(), color='r')
plt.show()
```

![scatter_bream_horizontal](https://github.com/seyong2/seyong2.github.io/blob/master/_posts/figures/scatter_bream_horizontal.png)


We can measure how well this horizontal line fits the data by calculating the total distance between the line and the data points. However, when the data point is above the line, the distance is negative, which makes the overall fit appear better than it really is. Thus, we compute sum of squared residuals (SSR) by squaring the distances and summing them up.

```
def SSR(y, y_hat):
    err = 0
    for i in range(len(y)):
        err += (y[i] - y_hat[i])**2
    return err

SSR(y, [y.mean()]*len(y))
```

The SSR for the horizontal line is 1488078.9714285715. 
By rotating the line, we can obtain a line (intercept and slope) whose sum of squared residuals is the smallest. However, if we rotate too much, the fit gets worse again so we need to find the sweet spot in-between. At the sweet spot, the function SSR will have no slope.

```
reg = LinearRegression()
reg.fit(np.array(x).reshape((-1, 1)), y)
y_hat = reg.intercept_ + reg.coef_*x
sns.scatterplot(x=x, y=y)
sns.lineplot(x=x, y=y_hat, color='r')
plt.show()
```

```
SSR(y, y_hat)
```
We see that the new line with least squares estimates fit much better than the horizontal line equal to the average weight.
