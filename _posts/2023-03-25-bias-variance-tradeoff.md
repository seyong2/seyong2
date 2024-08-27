---
layout: post
title: Bias-variance Tradeoff
subtitle: 
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, bias-variance tradeoff]
comments: true
---

Recently, on a flight to Zurich for the weekend, my husband posed an intriguing question.

"Do you remember how mean squared error (MSE) can be broken down into squared bias and variance?"

"Absolutely," I replied. "I just reviewed it a few days ago. Let me derive it for you."

But as soon as I picked up the pencil, confident I could walk him through the formula, my mind went blank. I couldn't recall what I thought I had mastered. What a disaster!

After returning from Zurich, I decided to revisit this concept, reminding myself that it's one of the core principles in machine learning.

The bias-variance trade-off is a fundamental idea in machine learning that explains the relationship between bias, variance, and the overall error in a predictive model. It illustrates how the errors in a model's predictions stem from two main sources—bias and variance—and how these two components interact.

1. **Bias**
- Bias refers to the error introduced by approximating a real-world problem, which may be complex, by a simplified model.
- High bias occurs when a model makes strong assumptions about the data, potentially leading to *underfitting*. Underfitting happens when the model is too simple to capture the underlying structure of the data, leading to systematic errors in predictions.
- For example, if the true relationship in the data were non-linear, a curve, a linear regression model would have a high bias.

2. **Variance**
- Variance refers to the error introduced by the model's sensitivity to the specific data set it was trained on.
- High variance occurs when a model is too complex, allowing it to fit the training data very closely (including noise). This can lead to *overfitting*, where the model performs well on the training data but poorly on unseen data.
- For example, a deep neural network might have high variance if it captures noise in the training data as if it were a true signal.

Therefore, it is ideal to have an algorithm that has both low bias and low variance. Unfortunately, bias and variance are inversely related, meaning that it is impossible to achieve a model with a low bias and a low variance at the same time (see figure below). On the one hand, complex models can capture a wide variety of patterns in the data, which reduces bias but can increase variance due to overfitting. On the other hand, simple models may not capture all the patterns in the data, leading to higher bias, but they are less likely to overfit, resulting in lower variance.

![image](https://github.com/user-attachments/assets/075ed449-fda1-4987-9b4a-377717d0c5d4)

Source: http://scott.fortmann-roe.com/docs/BiasVariance.html

Now that we understand the bias-variance trade-off, let's break down Mean Squared Error (MSE) into its components: variance and squared bias.

$$MSE = E[(y - \hat{y})^2]$$ 

$$= E[( y - E(\hat{y}) + E(\hat{y}) - \hat{y} )^2 ]$$

$$= E[ ( y - E(\hat{y}) )^2 + 2( y - E(\hat{y}) )( E(\hat{y}) - \hat{y} ) + ( E(\hat{y}) - \hat{y} )^2 ]$$

$$= E[ ( y - E(\hat{y}) )^2 ] + E[ ( E(\hat{y}) - \hat{y} )^2 ]$$

$$= \text{Bias}^2 + \text{Variance}$$

In summary, the bias-variance trade-off is about finding the right balance. The objective is to reduce both bias and variance to minimize the total error on unseen data. Effectively navigating this trade-off is crucial for developing models that perform well on new, unobserved data.
