```python
# load packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import kstest, norm

from imblearn.under_sampling import NearMiss, RandomUnderSampler

# To ignore warinings
import warnings
warnings.filterwarnings('ignore')
```

# 1. Understand the Context and Data Collection

## Define the objective
The objective of this competition is to predict which customers respond positively to an automobile insurance offer.

## Load Data
For this purpose, we load the necessary data from the corresponding competition in Kaggle ([link](https://www.kaggle.com/competitions/playground-series-s4e7/overview)).


```python
# load the training and test dataset
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
```

## Understand the Data
The dataset for this competition (both train and test) was generated from a deep learning model trained on the [Health Insurance Cross Sell Prediction Data dataset](https://www.kaggle.com/datasets/annantkumarsingh/health-insurance-cross-sell-prediction-data/data).

# 2. Data Cleaning

## Handle Missing Values
We'll now identify any missing data and decide how to handle it, whether by removal or imputation.


```python
# check whether the training dataset has missing observations
print(df_train.isna().sum())
print("\n")
print(df_test.isna().sum())
```

    id                      0
    Gender                  0
    Age                     0
    Driving_License         0
    Region_Code             0
    Previously_Insured      0
    Vehicle_Age             0
    Vehicle_Damage          0
    Annual_Premium          0
    Policy_Sales_Channel    0
    Vintage                 0
    Response                0
    dtype: int64
    
    
    id                      0
    Gender                  0
    Age                     0
    Driving_License         0
    Region_Code             0
    Previously_Insured      0
    Vehicle_Age             0
    Vehicle_Damage          0
    Annual_Premium          0
    Policy_Sales_Channel    0
    Vintage                 0
    dtype: int64
    

The training and test data don't have any missing observations.

## Remove Duplicates
We'll now identify and remove duplicate records.


```python
# check whether there are duplicated observations
print(df_train.duplicated().sum())
print("\n")
print(df_test.duplicated().sum())
```

    0
    
    
    0
    

There are no duplicated observations in both datasets.

## Correct Errors
We now fix any errors or inconsistencies in the data (e.g., incorrect data types, out-of-range values).


```python
# show the first five rows of df_train
df_train.head()
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
      <th>id</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Driving_License</th>
      <th>Region_Code</th>
      <th>Previously_Insured</th>
      <th>Vehicle_Age</th>
      <th>Vehicle_Damage</th>
      <th>Annual_Premium</th>
      <th>Policy_Sales_Channel</th>
      <th>Vintage</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Male</td>
      <td>21</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>Yes</td>
      <td>65101.0</td>
      <td>124.0</td>
      <td>187</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Male</td>
      <td>43</td>
      <td>1</td>
      <td>28.0</td>
      <td>0</td>
      <td>&gt; 2 Years</td>
      <td>Yes</td>
      <td>58911.0</td>
      <td>26.0</td>
      <td>288</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Female</td>
      <td>25</td>
      <td>1</td>
      <td>14.0</td>
      <td>1</td>
      <td>&lt; 1 Year</td>
      <td>No</td>
      <td>38043.0</td>
      <td>152.0</td>
      <td>254</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Female</td>
      <td>35</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>1-2 Year</td>
      <td>Yes</td>
      <td>2630.0</td>
      <td>156.0</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Female</td>
      <td>36</td>
      <td>1</td>
      <td>15.0</td>
      <td>1</td>
      <td>1-2 Year</td>
      <td>No</td>
      <td>31951.0</td>
      <td>152.0</td>
      <td>294</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# print the number of rows and columns in df_train and df_test
print("The training data have {} rows and {} columns.".format(df_train.shape[0], df_train.shape[1]))
print("\nThe test data have {} rows and {} columns.".format(df_test.shape[0], df_test.shape[1]))
```

    The training data have 11504798 rows and 12 columns.
    
    The test data have 7669866 rows and 11 columns.
    


```python
# prints information about df_train
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11504798 entries, 0 to 11504797
    Data columns (total 12 columns):
     #   Column                Dtype  
    ---  ------                -----  
     0   id                    int64  
     1   Gender                object 
     2   Age                   int64  
     3   Driving_License       int64  
     4   Region_Code           float64
     5   Previously_Insured    int64  
     6   Vehicle_Age           object 
     7   Vehicle_Damage        object 
     8   Annual_Premium        float64
     9   Policy_Sales_Channel  float64
     10  Vintage               int64  
     11  Response              int64  
    dtypes: float64(3), int64(6), object(3)
    memory usage: 1.0+ GB
    

I'll convert $Region\_Code$ and $Policy\_Sales\_Channel$ to integer.


```python
df_train['Region_Code'] = df_train['Region_Code'].astype('int64')
df_test['Region_Code'] = df_test['Region_Code'].astype('int64')

df_train['Policy_Sales_Channel'] = df_train['Policy_Sales_Channel'].astype('int64')
df_test['Policy_Sales_Channel'] = df_test['Policy_Sales_Channel'].astype('int64')
```


```python
# get a quick overview of the numeric data
df_train.describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))
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
      <th>id</th>
      <th>Age</th>
      <th>Driving_License</th>
      <th>Region_Code</th>
      <th>Previously_Insured</th>
      <th>Annual_Premium</th>
      <th>Policy_Sales_Channel</th>
      <th>Vintage</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.15048e+07</td>
      <td>1.15048e+07</td>
      <td>1.15048e+07</td>
      <td>1.15048e+07</td>
      <td>1.15048e+07</td>
      <td>1.15048e+07</td>
      <td>1.15048e+07</td>
      <td>1.15048e+07</td>
      <td>1.15048e+07</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.7524e+06</td>
      <td>38.3836</td>
      <td>0.998022</td>
      <td>26.4187</td>
      <td>0.462997</td>
      <td>30461.4</td>
      <td>112.425</td>
      <td>163.898</td>
      <td>0.122997</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.32115e+06</td>
      <td>14.9935</td>
      <td>0.0444312</td>
      <td>12.9916</td>
      <td>0.498629</td>
      <td>16454.7</td>
      <td>54.0357</td>
      <td>79.9795</td>
      <td>0.328434</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2630</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.8762e+06</td>
      <td>24</td>
      <td>1</td>
      <td>15</td>
      <td>0</td>
      <td>25277</td>
      <td>29</td>
      <td>99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.7524e+06</td>
      <td>36</td>
      <td>1</td>
      <td>28</td>
      <td>0</td>
      <td>31824</td>
      <td>151</td>
      <td>166</td>
      <td>0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.6286e+06</td>
      <td>49</td>
      <td>1</td>
      <td>35</td>
      <td>1</td>
      <td>39451</td>
      <td>152</td>
      <td>232</td>
      <td>0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.15048e+07</td>
      <td>85</td>
      <td>1</td>
      <td>52</td>
      <td>1</td>
      <td>540165</td>
      <td>163</td>
      <td>299</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get a quick overview of the categorical data
df_train.describe(include='object')
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
      <th>Gender</th>
      <th>Vehicle_Age</th>
      <th>Vehicle_Damage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11504798</td>
      <td>11504798</td>
      <td>11504798</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Male</td>
      <td>1-2 Year</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>6228134</td>
      <td>5982678</td>
      <td>5783229</td>
    </tr>
  </tbody>
</table>
</div>



We don't see any inconsistencies in the training data.

# 3. Data Preparation

## Data Transformation
Transform data into appropriate formats (e.g., scaling, encoding categorical variables).


```python
# how many unique values Gender, Vehicle_Age and Vehicle_Damange have?
df_train[['Gender', 'Vehicle_Age', 'Vehicle_Damage']].nunique()
```




    Gender            2
    Vehicle_Age       3
    Vehicle_Damage    2
    dtype: int64




```python
# one-hot encode for Gender and Vehicle_Damage as they have 2 unique values
enc = OneHotEncoder(drop='if_binary', sparse_output=False, dtype=np.int64)

df_train['Gender'] = enc.fit_transform(df_train[['Gender']]) # 1 for male and 0 otherwise
df_test['Gender'] = enc.fit_transform(df_test[['Gender']])

df_train['Vehicle_Damage'] = enc.fit_transform(df_train[['Vehicle_Damage']]) # 1 for Yes and 0 otherwise
df_test['Vehicle_Damage'] = enc.fit_transform(df_test[['Vehicle_Damage']])
```


```python
# ordinal encoding for Vehicle_Age
enc = OrdinalEncoder(categories=[['< 1 Year', '1-2 Year', '> 2 Years']], dtype=np.int64)

df_train['Vehicle_Age'] = enc.fit_transform(df_train[['Vehicle_Age']])
df_test['Vehicle_Age'] = enc.fit_transform(df_test[['Vehicle_Age']])
```

# 4. Data Exploration

## Univariate Analysis

### Categorical Variables

#### $Gender$


```python
# display the frequency of each category to compare the counts across categories.
sns.countplot(data=df_train, x='Gender')
```




    <Axes: xlabel='Gender', ylabel='count'>




    
![png](output_21_1.png)
    



```python
# show the proportion of each category
sns.countplot(data=df_train, x='Gender', stat='percent')
```




    <Axes: xlabel='Gender', ylabel='percent'>




    
![png](output_22_1.png)
    


More than half of the observations are male and the rest (46 percent) are female.

#### $Driving\_License$


```python
# display the frequency of each category to compare the counts across categories.
sns.countplot(data=df_train, x='Driving_License')
```




    <Axes: xlabel='Driving_License', ylabel='count'>




    
![png](output_24_1.png)
    



```python
# show the proportion of each category
sns.countplot(data=df_train, x='Driving_License', stat='percent');
```


    
![png](output_25_0.png)
    



```python
df_train['Driving_License'].value_counts(normalize=True)
```




    Driving_License
    1    0.998022
    0    0.001978
    Name: proportion, dtype: float64



Almost all customers have driver license, which makes me think whether this variable will be useful for predicting which customers respond positively to an automobile insurance offer as it shows very little variability.

#### $Region\_Code$


```python
# display the frequency of each category to compare the counts across categories.
plt.figure(figsize=(10, 6))
sns.countplot(data=df_train, x='Region_Code')
plt.xticks(rotation=45);
```


    
![png](output_28_0.png)
    



```python
# show the proportion of each category
plt.figure(figsize=(10, 6))
sns.countplot(data=df_train, x='Region_Code', stat='percent');
plt.xticks(rotation=45);
```


    
![png](output_29_0.png)
    



```python
# Summary Statistics
region_mode = df_train['Region_Code'].mode()[0]
unique_regions = df_train['Region_Code'].nunique()

print(f"Mode: {region_mode}")
print(f"Number of Unique Regions: {unique_regions}")
```

    Mode: 28
    Number of Unique Regions: 53
    

The customers are located in 54 different regions and around 30 percent of them live in region 28.

#### $Previously\_Insured$


```python
# display the frequency of each category to compare the counts across categories.
sns.countplot(data=df_train, x='Previously_Insured')
```




    <Axes: xlabel='Previously_Insured', ylabel='count'>




    
![png](output_32_1.png)
    



```python
# show the proportion of each category
sns.countplot(data=df_train, x='Previously_Insured', stat='percent');
```


    
![png](output_33_0.png)
    



```python
df_train['Previously_Insured'].value_counts(normalize=True)
```




    Previously_Insured
    0    0.537003
    1    0.462997
    Name: proportion, dtype: float64



This variable is more or less evenly distributed but there are 7 percent more customers who were not insured previously than those who were.

#### $Vehicle\_Age$


```python
# display the frequency of each category to compare the counts across categories.
sns.countplot(data=df_train, x='Vehicle_Age');
```


    
![png](output_36_0.png)
    



```python
# show the proportion of each category
sns.countplot(data=df_train, x='Vehicle_Age', stat='percent');
```


    
![png](output_37_0.png)
    



```python
df_train['Vehicle_Age'].value_counts(normalize=True)
```




    Vehicle_Age
    1    0.520016
    0    0.438438
    2    0.041546
    Name: proportion, dtype: float64



More than 95 percent of customers own a vehicle that is 2 years old or newer.

#### $Vehicle\_Damage$


```python
# display the frequency of each category to compare the counts across categories.
sns.countplot(data=df_train, x='Vehicle_Damage');
```


    
![png](output_40_0.png)
    



```python
# show the proportion of each category
sns.countplot(data=df_train, x='Vehicle_Damage', stat='percent');
```


    
![png](output_41_0.png)
    


Half of the customers have their vehicle with damages.

#### $Policy\_Sales\_Channel$


```python
# display the frequency of each category to compare the counts across categories.
plt.figure(figsize=(10, 6))
sns.countplot(data=df_train, x='Policy_Sales_Channel');
plt.xticks(rotation=45);
```


    
![png](output_43_0.png)
    



```python
# show the proportion of each category
plt.figure(figsize=(10, 6))
sns.countplot(data=df_train, x='Policy_Sales_Channel', stat='percent');
plt.xticks(rotation=45);
```


    
![png](output_44_0.png)
    



```python
# Summary Statistics
channel_mode = df_train['Policy_Sales_Channel'].mode()[0]
unique_channels = df_train['Policy_Sales_Channel'].nunique()

print(f"Mode: {channel_mode}")
print(f"Number of Unique Regions: {unique_channels}")
```

    Mode: 152
    Number of Unique Regions: 152
    

There are 152 unique channels where insurance policies are sold and the channel 152 is the most observed channel in the data. Would this variable be related to $Region\_Code$?

#### $Response$


```python
# display the frequency of each category to compare the counts across categories.
sns.countplot(data=df_train, x='Response');
```


    
![png](output_47_0.png)
    



```python
# show the proportion of each category
sns.countplot(data=df_train, x='Response', stat='percent');
```


    
![png](output_48_0.png)
    


Lastly, the dependent variable, $Response$, has an unbalanced distribution. The proportion of customers who responded negatively to an automobile insurance offer is much greater than the proportion of those who responded positively.

As you may already know, imbalanced classifications could result in models that have poor predictive performance, specifically for the minority class. In this competition, we are also more interested in classifying correctly positive cases (respond positively to an automobile insurance offer). Thus, we'll have to deal with this issue later before modeling.

We are now going to explore the continuous variables.

### Continuous Variables

#### $Age$


```python
# the frequency distribution of a numeric variable
sns.histplot(data=df_train, x='Age');
```


    
![png](output_50_0.png)
    



```python
# combine the distribution shape with summary statistics
sns.violinplot(data=df_train, x='Age')
```




    <Axes: xlabel='Age'>




    
![png](output_51_1.png)
    



```python
# check if the data follows a normal distribution.
sm.qqplot(df_train['Age'], line ='45');
```


    
![png](output_52_0.png)
    


The variable $Age$ is not normally distributed, with most customers being between 20 and 30 years old. According to the violin plot, there appear to be no outliers.

#### $Annual\_Premium$


```python
# the frequency distribution of a numeric variable
sns.histplot(data=df_train, x='Annual_Premium');
```


    
![png](output_54_0.png)
    



```python
# combine the distribution shape with summary statistics
sns.violinplot(data=df_train, x='Annual_Premium')
```




    <Axes: xlabel='Annual_Premium'>




    
![png](output_55_1.png)
    



```python
# check if the data follows a normal distribution.
sm.qqplot(df_train['Annual_Premium'], line ='45');
```


    
![png](output_56_0.png)
    


The distribution of $Annual\_Premium$ also seems to be far from a normal distribution. Furthermore, its distribution is right skewed meaning that it has a very long right tail. Thus, it is highly likely that there are outliers, which we'll identify later.

#### $Vintage$

This variable represents the number of days a customer has been associated with the insurance company.


```python
# the frequency distribution of a numeric variable
sns.histplot(data=df_train, x='Vintage');
```


    
![png](output_58_0.png)
    



```python
# combine the distribution shape with summary statistics
sns.violinplot(data=df_train, x='Vintage')
```




    <Axes: xlabel='Vintage'>




    
![png](output_59_1.png)
    



```python
# check if the data follows a normal distribution.
sm.qqplot(df_train['Vintage'], line ='45');
```


    
![png](output_60_0.png)
    


This variable seems to follow a uniform distribution as all the values are more or less equally likely.

https://www.kaggle.com/code/khangtran94vn/khang-eda-classification-insurance/notebook#Relationship-between-columns

## Multivariate Analysis


```python
plt.figure(figsize=(10, 10))
sns.heatmap(df_train.corr(), annot=True)
```




    <Axes: >




    
![png](output_64_1.png)
    


Examining the correlation between the variables...

- There is a strong positive correlation between $Age$ and $Vehicle\_Age$. This suggests that younger drivers tend to have newer vehicles, whereas older drivers are more likely to own older vehicles. The reason behind this would be that younger drivers prefer to own newer models, while older drivers keep their vehicles longer.
- $Previously\_Insured$ and $Vehicle\_Damage$ are strongly and negatively correlated. This implies that drivers who are not insured are more likely to have experienced vehicle damage in the past. This may be because those who were insured were being more cautious or having less risky driving behavior.
- $Previously\_Insured$ is also negatively correlated with $Vehicle\_Age$ and this is because $Vehicle\_Age$ and $Vehicle\_Damage$ are positively related; the newer the vehicle, the less likely that the vehicle has experienced damages.
- There is also a negative correlation between $Age$ and $Policy\_Sales\_Channel$. This means that different channels are used to reach out to the customers depending on their age. Older customers may favor traditional methods like agents or phone, while younger customers are more likely to use digital channels.
- As $Age$ and $Vehicle\_Age$ are strongly and positively correlated, $Policy\_Sales\_Channel$ is also negatively correlated with $Vehicle\_Age$.
- The dependent variable, $Response$, is negatively correlated with $Previously\_Insured$ as it is quite obvious that if you're already insured, you are more likely to respond negatively to insurance offers. In addition, $Response$ is positively related to $Vehicle\_Damage$. This may be due to the fact that if you have experienced damages to your vehicle in the past, you might want to have your vehicle insured.


### $Age$ and $Vehicle\_Age$


```python
sns.boxplot(x='Vehicle_Age', y='Age', data=df_train)
plt.show()
```


    
![png](output_66_0.png)
    



```python
sns.histplot(data=df_train, x='Age', hue='Vehicle_Age', palette=['red', 'blue', 'green']);
```


    
![png](output_67_0.png)
    


As we've seen earlier in the correlation heatmap, the plots indicate that vehicles less than 1 year old are typically owned by drivers under 35, while older vehicles are owned by older drivers.

### $Previously\_Insured$ and $Vehicle\_Damage$


```python
sns.countplot(x='Previously_Insured', hue='Vehicle_Damage', data=df_train);
```


    
![png](output_69_0.png)
    


We've already seen before that the drivers who were previously insured are more likely to have experience damages on their vehicles.


```python
g = sns.FacetGrid(df_train, col="Previously_Insured", row="Vehicle_Damage", margin_titles=True)
g.map(sns.countplot, "Response")
plt.show()
```


    
![png](output_71_0.png)
    


The majority of clients who respond positively to insurance offers are drivers who are not insured and have had their vehicles damaged in the past.

### $Age$ and $Policy\_Sales\_Channel$


```python
top7_policy_sales_channels = df_train['Policy_Sales_Channel'].value_counts()[:7].index
df_train_top7_policy_sales_channels = df_train[df_train['Policy_Sales_Channel'].isin(top7_policy_sales_channels)]
```


```python
sns.boxplot(x='Policy_Sales_Channel', y='Age', data=df_train_top7_policy_sales_channels)
plt.show()
```


    
![png](output_74_0.png)
    


Channel 152 and 160 are the ones that reach out to young clients, whereas the other target older audiences.

# 5. Model Training

Now we turn our attention to predicting whether a client responds negatively or positively, I'll build an artificial neural network model for classification. But before that, I will normalize the input features (except binary variables) as if we feed unnormalized inputs to activation functions, we can get stuck in a very flat region in the domain and may not learn at all. Or worse, we can end up with numerical issues.

## Scaling Data


```python
# normalize continuous features
scaler = MinMaxScaler()

df_train['Age'] = scaler.fit_transform(df_train[['Age']])
df_test['Age'] = scaler.transform(df_test[['Age']])

df_train['Annual_Premium'] = scaler.fit_transform(df_train[['Annual_Premium']])
df_test['Annual_Premium'] = scaler.transform(df_test[['Annual_Premium']])

df_train['Vintage'] = scaler.fit_transform(df_train[['Vintage']])
df_test['Vintage'] = scaler.transform(df_test[['Vintage']])

# normalize continuous features
df_train['Region_Code'] = scaler.fit_transform(df_train[['Region_Code']])
df_test['Region_Code'] = scaler.fit_transform(df_test[['Region_Code']])

df_train['Vehicle_Age'] = scaler.fit_transform(df_train[['Vehicle_Age']])
df_test['Vehicle_Age'] = scaler.fit_transform(df_test[['Vehicle_Age']])

df_train['Policy_Sales_Channel'] = scaler.fit_transform(df_train[['Policy_Sales_Channel']])
df_test['Policy_Sales_Channel'] = scaler.fit_transform(df_test[['Policy_Sales_Channel']])
```

## Splitting Data
Split the training data into independent and dependent datasets.


```python
y = df_train['Response']
X = df_train.drop(['Response', 'id'], axis=1)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Build ANN


```python
class LinearLayer:
    """
        This Class implements all functions to be executed by a linear layer
        in a computational graph
        Args:
            input_shape: input shape of Data/Activations
            n_out: number of neurons in layer
            ini_type: initialization type for weight parameters, default is "plain"
                      Opitons are: plain, xavier and he
        Methods:
            forward(A_prev)
            backward(upstream_grad)
            update_params(learning_rate)
    """

    def __init__(self, input_shape, n_out, ini_type="plain"):
        """
        The constructor of the LinearLayer takes the following parameters
        Args:
            input_shape: input shape of Data/Activations
            n_out: number of neurons in layer
            ini_type: initialization type for weight parameters, default is "plain"
        """

        self.m = input_shape[1]  # number of examples in training data
        # `params` store weights and bias in a python dictionary
        self.params = initialize_parameters(input_shape[0], n_out, ini_type)  # initialize weights and bias
        self.Z = np.zeros((self.params['W'].shape[0], input_shape[1]))  # create space for resultant Z output

    def forward(self, A_prev):
        """
        This function performs the forwards propagation using activations from previous layer
        Args:
            A_prev:  Activations/Input Data coming into the layer from previous layer
        """

        self.A_prev = A_prev  # store the Activations/Training Data coming in
        self.Z = np.dot(self.params['W'], self.A_prev) + self.params['b']  # compute the linear function

    def backward(self, upstream_grad):
        """
        This function performs the back propagation using upstream gradients
        Args:
            upstream_grad: gradient coming in from the upper layer to couple with local gradient
        """

        # derivative of Cost w.r.t W
        self.dW = np.dot(upstream_grad, self.A_prev.T)

        # derivative of Cost w.r.t b, sum across rows
        self.db = np.sum(upstream_grad, axis=1, keepdims=True)

        # derivative of Cost w.r.t A_prev
        self.dA_prev = np.dot(self.params['W'].T, upstream_grad)

    def update_params(self, learning_rate=0.1):
        """
        This function performs the gradient descent update
        Args:
            learning_rate: learning rate hyper-param for gradient descent, default 0.1
        """

        self.params['W'] = self.params['W'] - learning_rate * self.dW  # update weights
        self.params['b'] = self.params['b'] - learning_rate * self.db  # update bias(es)
```


```python
class SigmoidLayer:
    """
    This file implements activation layers
    inline with a computational graph model
    Args:
        shape: shape of input to the layer
    Methods:
        forward(Z)
        backward(upstream_grad)
    """

    def __init__(self, shape):
        """
        The consturctor of the sigmoid/logistic activation layer takes in the following arguments
        Args:
            shape: shape of input to the layer
        """
        self.A = np.zeros(shape)  # create space for the resultant activations

    def forward(self, Z):
        """
        This function performs the forwards propagation step through the activation function
        Args:
            Z: input from previous (linear) layer
        """
        self.A = 1 / (1 + np.exp(-Z))  # compute activations

    def backward(self, upstream_grad):
        """
        This function performs the  back propagation step through the activation function
        Local gradient => derivative of sigmoid => A*(1-A)
        Args:
            upstream_grad: gradient coming into this layer from the layer above
        """
        # couple upstream gradient with local gradient, the result will be sent back to the Linear layer
        self.dZ = upstream_grad * self.A*(1-self.A)

```


```python
def initialize_parameters(n_in, n_out, ini_type='plain'):
    """
    Helper function to initialize some form of random weights and Zero biases
    Args:
        n_in: size of input layer
        n_out: size of output/number of neurons
        ini_type: set initialization type for weights
    Returns:
        params: a dictionary containing W and b
    """

    params = dict()  # initialize empty dictionary of neural net parameters W and b

    if ini_type == 'plain':
        params['W'] = np.random.randn(n_out, n_in) *0.01  # set weights 'W' to small random gaussian
    elif ini_type == 'xavier':
        params['W'] = np.random.randn(n_out, n_in) / (np.sqrt(n_in))  # set variance of W to 1/n
    elif ini_type == 'he':
        # Good when ReLU used in hidden layers
        # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        # Kaiming He et al. (https://arxiv.org/abs/1502.01852)
        # http: // cs231n.github.io / neural - networks - 2 /  # init
        params['W'] = np.random.randn(n_out, n_in) * np.sqrt(2/n_in)  # set variance of W to 2/n

    params['b'] = np.zeros((n_out, 1))    # set bias 'b' to zeros

    return params

```


```python
def compute_cost(Y, Y_hat):
    """
    This function computes and returns the Cost and its derivative.
    The is function uses the Squared Error Cost function -> (1/2m)*sum(Y - Y_hat)^.2
    Args:
        Y: labels of data
        Y_hat: Predictions(activations) from a last layer, the output layer
    Returns:
        cost: The Squared Error Cost result
        dY_hat: gradient of Cost w.r.t the Y_hat
    """
    m = Y.shape[1]

    cost = (1 / (2 * m)) * np.sum(np.square(Y - Y_hat))
    cost = np.squeeze(cost)  # remove extraneous dimensions to give just a scalar

    dY_hat = -1 / m * (Y - Y_hat)  # derivative of the squared error cost function

    return cost, dY_hat
```


```python
def compute_stable_bce_cost(Y, Z):
    """
    This function computes the "Stable" Binary Cross-Entropy(stable_bce) Cost and returns the Cost and its
    derivative w.r.t Z_last(the last linear node) .
    The Stable Binary Cross-Entropy Cost is defined as:
    => (1/m) * np.sum(max(Z,0) - ZY + log(1+exp(-|Z|)))
    Args:
        Y: labels of data
        Z: Values from the last linear node
    Returns:
        cost: The "Stable" Binary Cross-Entropy Cost result
        dZ_last: gradient of Cost w.r.t Z_last
    """
    m = Y.shape[1]

    cost = (1/m) * np.sum(np.maximum(Z, 0) - Z*Y + np.log(1+ np.exp(- np.abs(Z))))
    dZ_last = (1/m) * ((1/(1+np.exp(- Z))) - Y)  # from Z computes the Sigmoid so P_hat - Y, where P_hat = sigma(Z)

    return cost, dZ_last
```

## Train the Model


```python
# define training constants
learning_rate = 1
number_of_epochs = 500

np.random.seed(48) # set seed value so that the results are reproduceable
                  # (weights will now be initailzaed to the same pseudo-random numbers, each time)


# Our network architecture has the shape: 
#                   (input)--> [Linear->Sigmoid] -> [Linear->Sigmoid] -->(output)  

#------ LAYER-1 ----- define hidden layer that takes in training data 
Z1 = LinearLayer(input_shape=(10, 2263750), n_out=10, ini_type='plain')
A1 = SigmoidLayer(Z1.Z.shape)

#------ LAYER-2 ----- define output layer that take is values from hidden layer
Z2 = LinearLayer(input_shape=A1.A.shape, n_out=1, ini_type='plain')
A2 = SigmoidLayer(Z2.Z.shape)
```


```python
train_costs = [] # initially empty list, this will store all the costs after a certian number of epochs
test_costs = []

# Set up the undersampling method
undersampler = RandomUnderSampler()

# Start training
for epoch in range(number_of_epochs):

    # Apply the transformation to the dataset
    X_train_sampled, y_train_sampled = undersampler.fit_resample(X_train, y_train)
    X_train_sampled = np.array(X_train_sampled).T
    y_train_sampled = np.array(y_train_sampled).reshape(-1, 1).T
    
    # ------------------------- forward-prop -------------------------
    Z1.forward(X_train_sampled)
    A1.forward(Z1.Z)
    
    Z2.forward(A1.A)
    A2.forward(Z2.Z)
    
    # ---------------------- Compute Cost ----------------------------
    train_cost, dZ2 = compute_stable_bce_cost(y_train_sampled, Z2.Z)
    train_costs.append(train_cost)
    
    # ------------------------- back-prop ----------------------------
    Z2.backward(dZ2)

    A1.backward(Z2.dA_prev)
    Z1.backward(A1.dZ)

    # ---------------------- Forward pass on the test data -----------
    Z1.forward(np.array(X_test).T)
    A1.forward(Z1.Z)
    
    Z2.forward(A1.A)
    A2.forward(Z2.Z)
    
    test_cost, dZ_last = compute_stable_bce_cost(np.array(y_test).reshape(-1, 1).T, Z2.Z)
    test_costs.append(test_cost)
    
    # print and store Costs every 100 iterations.
    if (epoch % 100) == 0:

        print("Cost at epoch# {}: on the training data - {}, on the test data - {}".format(epoch, train_cost, test_cost))

    # ----------------------- Update weights and bias ----------------
    Z2.update_params(learning_rate=learning_rate)
    Z1.update_params(learning_rate=learning_rate)
```

    Cost at epoch# 0: on the training data - 0.6931342992576633, on the test data - 0.695047958701558
    Cost at epoch# 100: on the training data - 0.5269429772504229, on the test data - 0.5486524265408564
    Cost at epoch# 200: on the training data - 0.4444701708273118, on the test data - 0.49260638677894864
    Cost at epoch# 300: on the training data - 0.44071909042647595, on the test data - 0.48967322066381624
    Cost at epoch# 400: on the training data - 0.43906724946782094, on the test data - 0.48747623761542985
    

### Evaluate the Model
We will now observe the cost on both the training and test data after each epoch. It can be seen that after approximately 150 epochs, the costs begin to stabilize and no longer decrease significantly. Of course, the cost on the test data is always lower than on the training data, as the model is trained using the training data, not the test data.


```python
plt.plot(train_costs, label='Training Cost')
plt.plot(test_costs, label='Test Cost')
plt.legend()
plt.show();
```


    
![png](output_91_0.png)
    


We will also examine the accuracy of the model's predictions on both the training and test data.


```python
## accuracy on the training data
# forward pass
Z1.forward(X_train.T)
A1.forward(Z1.Z)
Z2.forward(A1.A)
A2.forward(Z2.Z)

y_train_preds = np.round(A2.A).flatten().astype('int64')
y_train_arr = np.array(y_train)
train_acc = np.sum(y_train_preds == y_train_arr) / len(y_train_arr)

print(f"The accuracy on the training data is: {np.round(100*train_acc, 2)}%.")

# accuracy on the test data
# forward pass
Z1.forward(X_test.T)
A1.forward(Z1.Z)
Z2.forward(A1.A)
A2.forward(Z2.Z)

y_test_preds = np.round(A2.A).flatten().astype('int64')
y_test_arr = np.array(y_test)
test_acc = np.sum(y_test_preds == y_test_arr) / len(y_test_arr)

print(f"The accuracy on the test data is: {np.round(100*test_acc, 2)}%.")
```

    The accuracy on the training data is: 64.0%.
    The accuracy on the test data is: 64.05%.
    

We also generate a confusion matrix and calculate the ROC AUC score to gain additional insights into the performance of the classification model from different perspectives.


```python
# confusion matrix
# cm = confusion_matrix(y_test_arr, y_test_preds)
ConfusionMatrixDisplay.from_predictions(y_test_arr, y_test_preds, values_format='.0f')
plt.show()

# roc auc score
print(f"ROC AUC Score: {roc_auc_score(y_test_arr, y_test_preds)}")
```


    
![png](output_95_0.png)
    


    ROC AUC Score: 0.7869794981510511
    

From the confusion matrix, we can compute other interesting metrics to evaluate the model's performance.

- Precision = Proportion of true positives over total number of samples predicted as positive. In this example, the precision equals 0.25.
- Recall = Proportion of true positives over total number of samples that are actually positive. Here, we have a recall of 0.98.

As you may already know, there is a trade-off between precision and recall. Improving precision usually leads to a decrease in recall and vice versa. For example, if a model is tuned to be more conservative in making positive predictions to increase precision, it might miss more true positive cases, thus lowering recall. Conversely, if the model is tuned to identify as many positive instances as possible, it may increase recall but also raise the number of false positives, lowering precision. This is an essential concept in assessing the performance of classification models, especially in cases where classes are imbalanced or the costs of false positives and false negatives vary greatly. In our scenario, missing a true positive (failing to identify conductors who would respond positively to an insurance offer) is more detrimental, so we prioritize higher recall over higher precision.


### Make Predictions on the Test Data

With our model ready, we'll now make predictions on the test data to evaluate how well it performs on unseen data.


```python
Z1.forward(np.array(df_test.drop(['id'], axis=1)).T)
A1.forward(Z1.Z)
Z2.forward(A1.A)
A2.forward(Z2.Z)
```


```python
submission = pd.DataFrame({'id': df_test['id'], 'Response': A2.A.flatten()})
submission.to_csv('insurance_nn.csv', index=False)
```

The score after submitting to Kaggle was 0.83352. Considering that the model I created was just a basic neural network, I'm quite pleased with the result. In the future, I’d like to experiment with more sophisticated models to see how much I can improve the performance.
