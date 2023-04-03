---
layout: post
title: Bias-variance Tradeoff
subtitle: 
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, bias-variance tradeoff]
comments: true
---

The other day, during the flight to Zurich to spend the weekend, my husband asked me a question. 

"Do you remember how mean squared error (MSE) can be decomposed into squared-bias and variance?"

"Of course, in fact I looked it up a few days ago. Let me derive it for you."

However, even if I was quite confident about deriving the formula, I failed to recollect what I thought I had learned in the moment I grabbed the pencil... What a disaster!

After the trip to Zurich, I decided to review this concept reminding myself that this was one of the basic and main concepts in machine learning. The first thing I did, therefore, was to [watch the video by StatQuest with Josh Starmer](https://www.youtube.com/watch?v=EuBBz3bI-aA), which is my go-to Youtube channel whenever comes to my mind a question related to statistics or machine learning.

According to the video, bias can be defined as the inability of a machine learning algorithm to capture the true relationship between variables. Then, if the true relationship were non-linear, a curve, for example, a linear regression model would have a high bias. That is because the curve can never be expressed using a straight line. Nevertheless, this made me think how we know we made a wrong assumption if we use machine learning algorithms because we do not know the true relationship... 

On the other hand, variance is an error which results from a model that is sensitive to fluctuations in the training set. Then, if the model fits on different training data are very different from one another, the model has a high variance. In other words, it is ovefitting by modeling the random noise in the training data.

Therefore, it is ideal to have an algorithm that has both bias and variance low. Unfortunately, bias and variance are inversely related, meaning that it is impossible achieve a model with a low bias and a low variance. To see why, let's assume that we want to approximate a non-linear relationship using two machine learning methods; a straight line and a squiggly line. As we discussed earlier, the straight line would have a high bias because it is not capable of replicating the non-linear shape. In fact, the squiggly line would do a better job and have a much lower bias than the straight line. However, the variance of the straight line would be much lower than that of the squiggly line because the latter one would capture the random noise in the training data, resulting in a poor performance on different data. In sum, the less (more) a model complexity is, the higher (lower) bias and lower (higher) variance. Then, what matters the most is to find the level of model complexity at which the sum of bias (or squared-bias) and variance is the lowest. 
