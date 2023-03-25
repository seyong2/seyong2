---
layout: post
title: Linear Discriminant Analysis
subtitle: Predict Whether or Not a User Clicked on Ad
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, linear discriminant analysis, lda]
comments: true
---

In today's post, we are going to see what linear discriminant analysis (LDA) is and how it works through an easy example. The data used in this example is the same as the one we used to illustrate logistic regression. For more information about the data, check out [Kaggle](https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad). Let's have a look at the scatter plot below where we have **Daily Internet Usage** and **Age** on the $x$- and $y$-axis, respectively, for both users who clicked on the ad or not. Seemingly, the consumers who use more internet on a daily basis are less likely to have clicked on the ad. On the other hand, the variable, **Age**, does not seem to separate the consumers who clicked on the add from those who did not. If we are interested in having a new axis based on the two variables, instead of two as we have now, how can it be drawn in order to separate best two types of consumers?

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df = pd.read_csv('advertising.csv')

sns.scatterplot(x='Daily Internet Usage', y='Age', hue='Clicked on Ad', data=df)
```

![scatter_plot](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_lda/scatterplot.png?raw=true)

This is when LDA comes in to solve the problem. It tries to reduce dimensionality of data in a way that separability among known categories is maximized. This is done by creating a new axis given information of variables according to two criteria. The first one is that the distance between means of data projected onto the new axis should be maximized. The other one is that the variation within each category (called scatter) has to be minimized. Then, if, in this example, we denote $d$ is the distance between the means of two categories after the data is projected onto the new axis, and $s_1^2$ and $s_0^2$ as the scatter of the group of the consumers who did not click on the ad and that of the people who did not, respectively, LDA maximizes $\frac{d^2}{s_1^2+s_2^2}$ to find the optimal axis. We are going to use the function **LinearDiscriminantAnalysis** from scikit-learn to this end.

```
clf = LinearDiscriminantAnalysis()
clf.fit(df.loc[:, ['Daily Internet Usage', 'Age']], df['Clicked on Ad'])

print(clf.intercept_, clf.coef_)
```

The classifier produces the intercept (11.41245521) and coefficients (-0.09551712, 0.1605331) with which we are allowed to compute the decision boundary; $\beta_0+\beta_1X_1+\beta_2X_2$=0, where $\beta_0$ is the intercept, $\beta_1$ and $\beta_2$ are the slope coefficients for $X_1$ (**Daily Internet Usage**) and $X_2$ (**Age**), respectively. As we want to plot the boundary, we have to solve for **Age** from which we obtain $X_2=-\frac{\beta_0}{\beta_2}-\frac{\beta_1}{\beta_2}X_1$. The figure below shows us the scatter plot that we saw above plus the decision boundary. According to the figure, although there are misclassifications, the new axis seems to separate well the two types of consumers.

```
x = np.arange(100,280)
y = -(clf.intercept_/clf.coef_[0,1])-(clf.coef_[0,0]/clf.coef_[0,1])*x

sns.scatterplot(x='Daily Internet Usage', y='Age', hue='Clicked on Ad', data=df)
sns.lineplot(x=x, y=y)
```

![scatter_boundary](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_lda/scatterplot_boundary.png?raw=true)
