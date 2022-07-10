---
layout: post
title: Simple Linear Regression 
subtitle: Predict Weight of Fish Species Using Height
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [test]
comments: true
---

In this post, we are going to estimate the weight of a fish species, Bream, using their height via a simple linear regression model. The description of the data set used here can be found in [Kaggle](https://www.kaggle.com/datasets/aungpyaeap/fish-market?resource=download).

# Python libraries and Data

We start by loading necessary Python libraries and the data for the post.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('fish.csv')
df.head()
```

The data has 7 characteristics of 159 fishes in the market. The description of the columns are as follows:

- *Species*: species name of fish
- *Weight*: weight of fish in g
- *Length1*: vertical length in cm
- *Length2*: diagonal length in cm
- *Length3*: cross length in cm
- *Height*: height in cm
- *Width*: diagonal width in cm

```
sns.scatterplot(x=df.loc[:, 'Height'], y=df.loc[:, 'Weight'], hue=df.loc[:, 'Species'])
```

As said earlier, since we are only interested in the species, Bream, we slice the data.

```
df_bream = df[df.loc[:, 'Species'] == 'Bream']
x = df_bream.loc[:, 'Height']
y = df_bream.loc[:, 'Weight']
sns.scatterplot(x=x, y=y)
```

We would like to add a line to the data to see the trend. But, what is the best line?
A horizontal line that cuts through the average weight is likely to be the worst line that one can have. However, this gives us an idea about finding the optimal line.

```
scatter = sns.scatterplot(x=x, y=y)
scatter.axhline(y.mean(), color='r')
plt.show()
```
We can measure how well this horizontal line fits the data by calculating the total distance between the line and the data points. However, when the data point is greater than the line, the distance is negative, which makes the overall fit appear better than it really is. Thus, we square every distance and sum them up. It is called sum of squared residuals.

```
def SSR(y, y_hat):
    err = 0
    for i in range(len(y)):
        err += (y[i] - y_hat[i])**2
    return err

SSR(y, [y.mean()]*len(y))
```

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
