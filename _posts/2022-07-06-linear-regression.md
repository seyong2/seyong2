---
layout: post
title: Predict Weight of Fish Species 
subtitle: Using Linear Regression in Python
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [test]
comments: true
---

In this post, I try to give you a breif summary of a linear regression model based on the book [**The Elements of Statistical Learning**](https://link.springer.com/book/10.1007/978-0-387-84858-7). While I walk you through the concepts, I will also provide a code demonstration using data from [Kaggle](https://www.kaggle.com/datasets/aungpyaeap/fish-market?resource=download) about common fish species in fish market. Then, the goal is to estimate the weight of seven common different fish species using a linear regression model. 

# Python libraries and Data

We load necessary Python libraries and the data for the post.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

The data has 7 characteristics of 159 fishes in the market. The description of the columns are as follows:

- *Species*: species name of fish
- *Weight*: weight of fish in g
- *Length1*: vertical length in cm
- *Length2*: diagonal length in cm
- *Length3*: cross length in cm
- *Height*: height in cm
- *Width*: diagonal width in cm

We clearly see that column *Species* is the only variable that is qualitative. Thus, we create dummy variables that are numeric variables to represent the sub-levels of the fish species of the data. As *Species* has seven different levels (Bream, Parkki, Perch, Pike, Roach, Smelt and Whitefish), the number of resulting dummies is six. 

```
ohe = OneHotEncoder(drop='first', sparse=False)
df_species = pd.DataFrame(ohe.fit_transform(df[['Species']]), columns=['Species 1','Species 2', 'Species 3', 'Species 4', 'Species 5', 'Species 6'])
df = pd.concat([df.iloc[:, 1:], df_species], axis=1)
df.head()
```

Now we divide the data into input matrix (*X*) and output (*y*) vector.
```
y = df.loc[:, 'Weight']
X = df.loc[:, df.columns != 'Weight']
```

Looking at the correlation matrix of the independent variables, we observe high positive correlations among *Length1*, *Length2*, *Length3*, *Height*, and *Width*.


**Here is some bold text**

## Here is a secondary heading

Here's a useless table:

| Number | Next number | Previous number |
| :------ |:--- | :--- |
| Five | Six | Four |
| Ten | Eleven | Nine |
| Seven | Eight | Six |
| Two | Three | One |


How about a yummy crepe?

![Crepe](https://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg)

It can also be centered!

![Crepe](https://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg){: .mx-auto.d-block :}

Here's a code chunk:

~~~
var foo = function(x) {
  return(x + 5);
}
foo(3)
~~~

And here is the same code with syntax highlighting:

```javascript
var foo = function(x) {
  return(x + 5);
}
foo(3)
```

And here is the same code yet again but with line numbers:

{% highlight javascript linenos %}
var foo = function(x) {
  return(x + 5);
}
foo(3)
{% endhighlight %}

## Boxes
You can add notification, warning and error boxes like this:

### Notification

{: .box-note}
**Note:** This is a notification box.

### Warning

{: .box-warning}
**Warning:** This is a warning box.

### Error

{: .box-error}
**Error:** This is an error box.
