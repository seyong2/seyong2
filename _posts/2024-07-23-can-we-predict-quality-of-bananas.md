---
layout: post
title: Can we predict quality of banana?
subtitle: Exploratory Data Analysis and Logistic Regression
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [Artificial Intelligence, Machine Learning, Deep Learning, Data Science, Logistic Regression]
comments: true
---

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/banana/banana_quality.csv
    

# Banana Quality Data

This [dataset](https://www.kaggle.com/datasets/l3llff/banana) consists of numerical characteristics of bananas of different quality (Good or Bad). We are going to explore the data and try to create a model that will predict the quality of bananas.


```python
# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Load the data



```python
df = pd.read_csv('/kaggle/input/banana/banana_quality.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Size</th>
      <th>Weight</th>
      <th>Sweetness</th>
      <th>Softness</th>
      <th>HarvestTime</th>
      <th>Ripeness</th>
      <th>Acidity</th>
      <th>Quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.924968</td>
      <td>0.468078</td>
      <td>3.077832</td>
      <td>-1.472177</td>
      <td>0.294799</td>
      <td>2.435570</td>
      <td>0.271290</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.409751</td>
      <td>0.486870</td>
      <td>0.346921</td>
      <td>-2.495099</td>
      <td>-0.892213</td>
      <td>2.067549</td>
      <td>0.307325</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.357607</td>
      <td>1.483176</td>
      <td>1.568452</td>
      <td>-2.645145</td>
      <td>-0.647267</td>
      <td>3.090643</td>
      <td>1.427322</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.868524</td>
      <td>1.566201</td>
      <td>1.889605</td>
      <td>-1.273761</td>
      <td>-1.006278</td>
      <td>1.873001</td>
      <td>0.477862</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.651825</td>
      <td>1.319199</td>
      <td>-0.022459</td>
      <td>-1.209709</td>
      <td>-1.430692</td>
      <td>1.078345</td>
      <td>2.812442</td>
      <td>Good</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (8000, 8)



There are 8 different information of 8,000 bananas.

## Understand the variables


```python
var_df = pd.DataFrame(columns=['Variable', 'Number of Unique Values', 'Values'])

for i, var in enumerate(df.columns):
    var_df.loc[i] = [var, df[var].nunique(), df[var].unique().tolist()]
var_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Number of Unique Values</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Size</td>
      <td>8000</td>
      <td>[-1.9249682, -2.4097514, -0.3576066, -0.868523...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Weight</td>
      <td>8000</td>
      <td>[0.46807805, 0.48686993, 1.4831762, 1.5662014,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sweetness</td>
      <td>8000</td>
      <td>[3.0778325, 0.34692144, 1.5684522, 1.8896049, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Softness</td>
      <td>8000</td>
      <td>[-1.4721768, -2.4950993, -2.6451454, -1.273761...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HarvestTime</td>
      <td>8000</td>
      <td>[0.2947986, -0.8922133, -0.64726734, -1.006277...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ripeness</td>
      <td>8000</td>
      <td>[2.4355695, 2.0675488, 3.0906434, 1.8730015, 1...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Acidity</td>
      <td>8000</td>
      <td>[0.27129033, 0.30732512, 1.427322, 0.47786173,...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Quality</td>
      <td>2</td>
      <td>[Good, Bad]</td>
    </tr>
  </tbody>
</table>
</div>



## Data Dictionary

* `Size` (float): Size of banana
* `Weight` (float): Weight of banana
* `Sweetness` (float): Sweetness of banana
* `Softness` (float): Softness of banana
* `HarvestTime` (float): Amount of time passed from harvesting of the fruit
* `Ripeness` (float): Ripeness of banana
* `Acidity` (float): Acidity of banana
* `Quality` (string): Quality of banana

## Interesting Questions
Now we will get to explore this exciting dataset! Let's try our hand at these questions:

* **Which characteristic would be the most important factor that determines the quality of banana?**
* **How are the characteristics related to each other? How do the characteristics of banana change as it ripens over time?**
* **Can we predict the quality of banana with this information?**

## Exploratory Data Analysis (EDA)
The dataset is not huge, we can easily get to know it and decide how to tackle the interesting questions that we defined above. From the introductory code above we know:

* Quality is a binary variable taking 2 values (Good and Bad).
* The other remaining variables are numerical continuous variables.

First of all, let's ensure there are no nulls:


```python
df.isna().sum()
```




    Size           0
    Weight         0
    Sweetness      0
    Softness       0
    HarvestTime    0
    Ripeness       0
    Acidity        0
    Quality        0
    dtype: int64



Indeed, there is no missing values in the dataset.


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe" width:"100%">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Size</th>
      <th>Weight</th>
      <th>Sweetness</th>
      <th>Softness</th>
      <th>HarvestTime</th>
      <th>Ripeness</th>
      <th>Acidity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.747802</td>
      <td>-0.761019</td>
      <td>-0.770224</td>
      <td>-0.014441</td>
      <td>-0.751288</td>
      <td>0.781098</td>
      <td>0.008725</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.136023</td>
      <td>2.015934</td>
      <td>1.948455</td>
      <td>2.065216</td>
      <td>1.996661</td>
      <td>2.114289</td>
      <td>2.293467</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-7.998074</td>
      <td>-8.283002</td>
      <td>-6.434022</td>
      <td>-6.959320</td>
      <td>-7.570008</td>
      <td>-7.423155</td>
      <td>-8.226977</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-2.277651</td>
      <td>-2.223574</td>
      <td>-2.107329</td>
      <td>-1.590458</td>
      <td>-2.120659</td>
      <td>-0.574226</td>
      <td>-1.629450</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.897514</td>
      <td>-0.868659</td>
      <td>-1.020673</td>
      <td>0.202644</td>
      <td>-0.934192</td>
      <td>0.964952</td>
      <td>0.098735</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.654216</td>
      <td>0.775491</td>
      <td>0.311048</td>
      <td>1.547120</td>
      <td>0.507326</td>
      <td>2.261650</td>
      <td>1.682063</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.970800</td>
      <td>5.679692</td>
      <td>7.539374</td>
      <td>8.241555</td>
      <td>6.293280</td>
      <td>7.249034</td>
      <td>7.411633</td>
    </tr>
  </tbody>
</table>
</div>



It seems that there are some non-ordinary values across all numrical continuous variables as we see negative minimum values and they do not seem to make sense given the meaning of the characteristics. However, we cannot exclude the possibility of them being treated/scaled already in some way, i.e., mean-centered, which would have converted some positive values into negative ones. Let's look at their distribution.


```python
num_cols = df.select_dtypes(include=np.number).columns

n_row, n_col = 2, 4
fig, axs = plt.subplots(n_row, n_col, figsize=[20, 12])

for column, ax in zip(num_cols, axs.ravel()):
    sns.histplot(data=df, x=column, kde=True, stat='density', ax=ax)

```   
![output_16_1](https://github.com/user-attachments/assets/55994a4a-4273-4e56-8f4f-7a1929c73333)

It appears that the variables underwent a kind of scaling, as we anticipated. Their distributions suggest that the values fall within a similar range and exhibit symmetrical patterns around zero.

**Which characteristic would be the most important factor that determines the quality of banana?**

To explore the relationship between banana quality and each characteristic, let's observe how the quality values vary for each variable between high-quality and low-quality bananas.


```python
n_row, n_col = 2, 4
fig, axs = plt.subplots(n_row, n_col, figsize=[20, 12])

for column, ax in zip(num_cols, axs.ravel()):
    sns.histplot(data=df, x=column, hue='Quality', stat='density', ax=ax)

plt.show()
```
![output_19_1](https://github.com/user-attachments/assets/8ef803ae-aa5e-4d2d-99a3-1ba19dedd906)
  
Upon comparing the distributions of each characteristic between good and bad quality bananas, distinct disparities emerge, with the exception of the `Acidity` variable, where the distributions exhibit some overlap. Nevertheless, we intend to proceed with $t$-tests to ascertain whether statistically significant differences exist between the distributions of the two types of bananas.

As you may already know, t-tests serve to assess if there exists a significant difference between the means of two groups. The $t$-value is calculated by dividing the difference between the two means by the uncertainty inherent in this difference, known as the standard error. This adjustment is necessary due to the inherent variability in sampled data, which inevitably introduces error into the determination of means.


```python
num_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in num_cols:
    good_qual_bananas = df.loc[df['Quality']=='Good', col]
    bad_qual_bananas = df.loc[df['Quality']=='Bad', col]
    print("The results of the t-test for " + col + ': ')
    result = stats.ttest_ind(good_qual_bananas, bad_qual_bananas)
    print(result)
    if result[1] < 0.05:
        print('Reject the null hypothesis; there is a significant difference between the sample means.')
    else:
        print('Fail to reject the null hypothesis; there is no significant difference between the sample means.')
    print('\n')
```

    The results of the t-test for Size: 
    TtestResult(statistic=33.73962998335712, pvalue=1.960547470682263e-233, df=7998.0)
    Reject the null hypothesis; there is a significant difference between the sample means.
    
    
    The results of the t-test for Weight: 
    TtestResult(statistic=37.739037902011866, pvalue=5.513984954490505e-287, df=7998.0)
    Reject the null hypothesis; there is a significant difference between the sample means.
    
    
    The results of the t-test for Sweetness: 
    TtestResult(statistic=36.42641264816277, pvalue=6.195977428863384e-269, df=7998.0)
    Reject the null hypothesis; there is a significant difference between the sample means.
    
    
    The results of the t-test for Softness: 
    TtestResult(statistic=-0.14644228329667378, pvalue=0.8835759406046255, df=7998.0)
    Fail to reject the null hypothesis; there is no significant difference between the sample means.
    
    
    The results of the t-test for HarvestTime: 
    TtestResult(statistic=36.35666232582134, pvalue=5.477675103055471e-268, df=7998.0)
    Reject the null hypothesis; there is a significant difference between the sample means.
    
    
    The results of the t-test for Ripeness: 
    TtestResult(statistic=33.469300228286556, pvalue=5.656660876948405e-230, df=7998.0)
    Reject the null hypothesis; there is a significant difference between the sample means.
    
    
    The results of the t-test for Acidity: 
    TtestResult(statistic=-0.07687842918995251, pvalue=0.9387221807334263, df=7998.0)
    Fail to reject the null hypothesis; there is no significant difference between the sample means.
    
    
    

At a significance level of 5 percent, our analysis suggests that, apart from softness and acidity, the mean values of the other variables are significantly different between the two quality types of bananas. To put it differently, those variables may play significant roles in determining the quality of bananas.

**How are the characteristics related to each other? How do the characteristics of banana change as it ripens over time?**

To examine the relationships between variables, we plot pairwise relationships between each pair of variables.


```python
sns.pairplot(df)
```
![output_25_2](https://github.com/user-attachments/assets/e441a00f-9bce-47f2-8d3b-daa89ad9b8ee)

```python
sns.pairplot(df, hue='Quality')
```
![output_26_2](https://github.com/user-attachments/assets/56bd135c-cabc-48f6-8760-146f9cfc322b)

Based on the depicted plots, it appears that there are no discernible relationships among the variables. This could be attributed to the fact that the variables have already been scaled, potentially causing any relationships to become indiscernible.

## Modeling

Now that we have examined the relationships among the variables, we will address our final question: **Can we predict the quality of bananas using this information?** To do this, we will construct a simple logistic model to predict banana quality, as the dependent variable is binary.

### Split the data into training and test datasets

First, we will split the data into training and test datasets. The training set will comprise 70% of the total data, while the remaining 30% will be assigned to the test set.


```python
# Define dependent and independent variables
y = df['Quality'].replace({'Good': 1, 'Bad': 0})
X = df.drop(['Quality'], axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```   

### Fit a logistic regression model on the training data


```python
# Fit a logistic regression model on the training data
model = LogisticRegression().fit(X_train, y_train)
```

As we've trained our logistic regression model on the training data, let's see how the model looks like.


```python
coefs = pd.DataFrame(model.coef_)
coefs.columns = model.feature_names_in_
coefs['Intercept'] = model.intercept_
coefs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Size</th>
      <th>Weight</th>
      <th>Sweetness</th>
      <th>Softness</th>
      <th>HarvestTime</th>
      <th>Ripeness</th>
      <th>Acidity</th>
      <th>Intercept</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.689277</td>
      <td>1.011169</td>
      <td>0.773299</td>
      <td>0.083834</td>
      <td>0.556131</td>
      <td>0.592052</td>
      <td>-0.104325</td>
      <td>1.72386</td>
    </tr>
  </tbody>
</table>
</div>



The training logistic regression is defined as:
$P(y=Good)=1 \over 1+e^{-(1.72+0.69 \times Size+1.01 \times Weight+0.77 \times Sweetness+0.08 \times Softness+0.56 \times HarvestTime+0.59 \times Ripeness-0.10 \times Acidity)}$

In order to interpret the coefficients, we should know first that for a feature $x_i$ with coefficient $\beta_i$, a one-unit increase in $x_i$ changes the log-odds by $\beta_i$. In other words, the corresponding change in odds is multiplicative by a factor of $e^{\beta_i}$. Then, for example, if the size of a banana increases by one unit, the odds of it being good quality increase by 1.99.

### Predict the quality of bananas in the test data
Now, we'll use this model to predict the quality of bananas in the test data and evaluate the accuracy of these predictions.


```python
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Logistic Regression model accuracy (in %):", acc*100)
```

    Logistic Regression model accuracy (in %): 87.72727272727273
    
