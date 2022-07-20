Today's post is about logistic regression that predicts a categorical dependent variable given a set of independent variables. We will go through important concepts of logistic regression with data about advertisement downloaded from [Kaggle](https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad?resource=download). The goal here is to predict well whether a user clicked on an advertisement or not using logistic regression. Let's start by importing the data and libraries needed throughout the post.  

```
```

![df_head]()

We see that the data has 1,000 rows and 10 columns. Each row represent information of a user and the ten columns describe the following:

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

The target variable here is **Clicked on Ad** which is a binary variable. As an easy example, we are going to estimate the relationship between **Clicked on Ad** and **Age**, a continuous measurement. The scatter plot below shows the values for the two variables. We see that the relationship cannot be estimated using linear regression. Instead, a *S*-shaped curve would be appropriate in this case. That is what logistic regression exactly does by fitting **logistic function**. The function takes a value between 0 and 1, representing the probability that a consumer clicked on the ad given his/her age. Then, if the probability exceeds a certain threshold value of 50 percent, for example, we classify that the consumer clicked on the ad.

To fit logistic function, we no longer minimize the sum of squared residuals (SSR) because the outcome can be 0 or 1 from which we cannot compute residuals. Then, we use maximum likelihood for estimating parameters of logistic function which maximizes the probability of obtaining observed data.
