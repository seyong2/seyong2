---
layout: post
title: Linear Discriminant Analysis
subtitle: Predict Whether or Not a User Clicked on Ad
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, linear discriminant analysis, lda]
comments: true
---

In today’s post, we'll explore Linear Discriminant Analysis (LDA) and demonstrate how it works with a straightforward example. We'll be using the same dataset as our previous discussion on logistic regression, which identifies whether or not a particular internet user clicked on an advertisement. If you want to learn more about the dataset, you can find it on [Kaggle](https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad).

Linear Discriminant Analysis (LDA) is a statistical technique widely used in machine learning and statistics to identify a linear combination of features that best separates two or more classes. It serves both as a dimensionality reduction tool and a classifier, making it particularly useful in applications like pattern recognition, face recognition, and other areas where accurate classification is essential.

To demonstrate how LDA works, let's examine the scatter plot below, which shows $Daily$ $Internet$ $Usage$ on the x-axis and $Age$ on the y-axis for users who either clicked on the ad or didn't. From the plot, it appears that users with higher daily internet usage are less likely to have clicked on the ad. The relationship between higher daily internet usage and the likelihood of clicking an ad can vary. Users with high internet usage might be less likely to click ads due to factors like ad fatigue, familiarity with online content, or focus on specific tasks. However, this isn't a universal trend, and the impact can depend on the relevance and quality of the ads, as well as the user's interests. The relationship is complex and should be validated with data. However, the variable Age does not seem to effectively distinguish between users who clicked on the ad and those who didn't.

Now, suppose we want to create a new axis that combines these two variables into a single one that maximizes the separation between the two types of consumers. How should we draw this new axis to best distinguish between those who clicked on the ad and those who didn’t?

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

This is where Linear Discriminant Analysis (LDA) comes into play to address the problem. LDA aims to reduce the dimensionality of the data while maximizing the separability between known categories. It achieves this by creating a new axis based on two key criteria:

1. **Maximizing the Distance Between Class Means**: LDA projects the data onto a new axis in such a way that the distance between the means of the different classes is maximized. This helps in distinguishing between the categories more effectively.

2. **Minimizing the Within-Class Variance**: At the same time, LDA minimizes the variation within each category (known as scatter) when the data is projected onto the new axis. This reduces the overlap between categories, further enhancing separability.

In this context, let's denote $d$ as the distance between the means of two categories after projection, and $s_1^2$ and $s_0^2$ as the scatter (variance) of the group that did not click on the ad and the group that did, respectively. LDA seeks to maximize the ratio $\frac{d^2}{s_1^2+s_2^2}$ to determine the optimal axis for projection.

To understand how to solve a problem using Linear Discriminant Analysis (LDA), let's walk through the mathematical steps in more detail. 

1. **Formulate the Problem**
   - Suppose you have a dataset with $n$ samples, each with $d$ features. The data is labeled into two classes, $C_1$ and $C_2$. Let $\mathbf{x}_i$ represent a feature vector (data point) and $y_i$ be the corresponding class label.

3. **Compute the Mean Vectors**
   - First, calculate the mean vector for each class:
     ```math
     \mathbf{\mu}_1 = \frac{1}{n_1} \sum_{\mathbf{x}_i \in C_1} \mathbf{x}_i, \quad \mathbf{\mu}_2 = \frac{1}{n_2} \sum_{\mathbf{x}_i \in C_2} \mathbf{x}_i
     ```
     where $n_1$ and $n_2$ are the number of samples in classes $C_1$ and $C_2$, respectively.

3. **Compute the Scatter Matrices**

   - **Within-Class Scatter Matrix ($\mathbf{S}_W$)**:
     ```math
     \mathsf{\mathbf{S}_W} = \sum_{\mathbf{x}_i \in C_1} (\mathbf{x}_i - \mathbf{\mu}_1)(\mathbf{x}_i - \mathbf{\mu}_1)^T + \sum_{\mathbf{x}_i \in C_2} (\mathbf{x}_i - \mathbf{\mu}_2)(\mathbf{x}_i - \mathbf{\mu}_2)^T
     ```
     $\mathbf{S}_W$ measures how much the samples within each class scatter around their mean.

   - **Between-Class Scatter Matrix ($\mathbf{S}_B$)**:
     ```math
     \mathbf{S}_B = (\mathbf{\mu}_1 - \mathbf{\mu}_2)(\mathbf{\mu}_1 - \mathbf{\mu}_2)^T
     ```
     $\mathbf{S}_B$ measures how much the means of the classes scatter with respect to each other.

4. **Compute the Optimal Projection Vector ($\mathbf{w}$)**
   - The goal of LDA is to find a projection vector $\mathbf{w}$ that maximizes the separability between the classes. This vector is found by solving the following optimization problem:
     ```math
     \mathbf{w} = \mathbf{S}_W^{-1} (\mathbf{\mu}_1 - \mathbf{\mu}_2)
     ```
     Here, $\mathbf{S}_W^{-1}$ is the inverse of the within-class scatter matrix, and $\mathbf{\mu}_1 - \mathbf{\mu}_2$ is the difference between the mean vectors of the two classes.

5. **Project the Data onto the New Axis**
   - Once the optimal $\mathbf{w}$ is found, you project each data point onto this vector:
     ```math
     z_i = \mathbf{w}^T \mathbf{x}_i
     ```
     This projection reduces the dimensionality of the data (in this case, to a single dimension) while maximizing the separation between the classes.

6. **Classification**
   - To classify a new data point $\mathbf{x}_{\text{new}}$, compute the projection $z_{\text{new}} = \mathbf{w}^T \mathbf{x}_{\text{new}}$.
   - A common approach is to use a threshold $\theta$ for classification:
     ```math
     \text{If } z_{\text{new}} > \theta \text{, classify as } C_1 \text{; otherwise, classify as } C_2.
     ```
   - The threshold $\theta$ can be chosen based on the midpoint between the projected means of the two classes:
     ```math
     \theta = \frac{1}{2} \mathbf{w}^T (\mathbf{\mu}_1 + \mathbf{\mu}_2)
     ```

7. **Generalization to Multiple Classes**
   - For more than two classes, LDA generalizes to finding a set of projection vectors (discriminants) that maximize class separability. The scatter matrices are defined similarly, but you solve a generalized eigenvalue problem:
     ```math
     \mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w}_i = \lambda_i \mathbf{w}_i
     ```
     where $\lambda_i$ are the eigenvalues and $\mathbf{w}_i$ are the corresponding eigenvectors. The eigenvectors corresponding to the largest eigenvalues are chosen as the discriminants.

In Python, we will use the **LinearDiscriminantAnalysis** function from scikit-learn to implement this approach.

```
clf = LinearDiscriminantAnalysis()
clf.fit(df.loc[:, ['Daily Internet Usage', 'Age']], df['Clicked on Ad'])

print(clf.intercept_, clf.coef_)
```

The classifier yields an intercept of 11.41 and coefficients of -0.0955 and 0.1605, which we can use to compute the decision boundary. The decision boundary is defined by the equation $\beta_0+\beta_1X_1+\beta_2X_2$=0, where $\beta_0$ is the intercept, and $\beta_1$ and $\beta_2$ are the coefficients corresponding to $X_1$ (**Daily Internet Usage**) and $X_2$ (**Age**), respectively. To plot the boundary, we solve for **Age** ($X_2$), yielding the equation $X_2=-\frac{\beta_0}{\beta_2}-\frac{\beta_1}{\beta_2}X_1$. The figure below shows the scatter plot, including the decision boundary. Despite some misclassifications, the plot indicates that the decision boundary effectively separates the two consumer groups.

```
x = np.arange(100,280)
y = -(clf.intercept_/clf.coef_[0,1])-(clf.coef_[0,0]/clf.coef_[0,1])*x

sns.scatterplot(x='Daily Internet Usage', y='Age', hue='Clicked on Ad', data=df)
sns.lineplot(x=x, y=y)
```

![scatter_boundary](https://github.com/seyong2/seyong2.github.io/blob/master/assets/img/figures_lda/scatterplot_boundary.png?raw=true)

The plot shows that the decision boundary effectively separates the two classes. However, it's important to be aware of some limitations when using Linear Discriminant Analysis (LDA). LDA assumes that the data follows a Gaussian distribution with identical covariance matrices for all classes. This assumption may not hold in many real-world scenarios. Additionally, LDA performs best when the classes are linearly separable or close to it. Another limitation is LDA's sensitivity to outliers, which can significantly distort the calculated means and covariance matrices, leading to less accurate predictions. I hope this post has helped you gain a clearer understanding of this statistical method.
