---
layout: post
title: Bias-variance Tradeoff
subtitle: 
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [machine learning, linear discriminant analysis, lda]
comments: true
---

The other day, during the flight to Zurich to spend the weekend, my husband asked me a question. 

"Do you remember how mean squared error (MSE) can be decomposed into squared-bias and variance?"

"Of course, in fact I looked it up a few days ago. Let me derive it for you."

However, even if I was quite confident about deriving the formula, I failed to recollect what I thought I had learned in the moment I grabbed the pencil... What a disaster!

After the trip to Zurich, I decided to review this concept reminding myself that this was one of the basic and main concepts in machine learning. The first thing I did, therefore, was to [watch the video by StatQuest with Josh Starmer](https://www.youtube.com/watch?v=EuBBz3bI-aA), which is my go-to Youtube channel whenever comes a question related to statistics or machine learning to my mind.

According to the video, bias is the inability of a machine learning algorithm to capture the true relationship between variables. Then, if the true relationship were non-linear, a curve, for example, a linear regression model would have a high bias. Nevertheless, this made me think that if we use machine learning algorithms because we do not know the true relationship, how do we know we made the wrong assumption?
