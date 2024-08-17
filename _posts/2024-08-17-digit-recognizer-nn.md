---
layout: post
title: Predicting Digits using Neural Network
subtitle: Simple Neural Network for Multi-Class Classification Problem
gh-repo: seyong2
gh-badge: [star, fork, follow]
tags: [Artificial Intelligence, Machine Learning, Data Science, Neural Network, Deep Learning]
comments: true
---

This notebook is designed to accurately recognize digits in a dataset containing tens of thousands of handwritten images, provided by the "Modified National Institute of Standards and Technology" (MNIST). To achieve this, we will build a simple neural network using PyTorch. For further information, refer to this [link](https://www.kaggle.com/competitions/digit-recognizer/overview).

Let's begin by importing the necessary libraries and loading the data.

# 1. Import Python libraries and Data


```python
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
%matplotlib inline

import random

from sklearn.preprocessing import OneHotEncoder

! pip install torchmetrics
import torch, torchmetrics
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
```

    Collecting torchmetrics
      Downloading torchmetrics-1.4.1-py3-none-any.whl.metadata (20 kB)
    Requirement already satisfied: numpy>1.20.0 in /usr/local/lib/python3.10/dist-packages (from torchmetrics) (1.26.4)
    Requirement already satisfied: packaging>17.1 in /usr/local/lib/python3.10/dist-packages (from torchmetrics) (24.1)
    Requirement already satisfied: torch>=1.10.0 in /usr/local/lib/python3.10/dist-packages (from torchmetrics) (2.3.1+cu121)
    Collecting lightning-utilities>=0.8.0 (from torchmetrics)
      Downloading lightning_utilities-0.11.6-py3-none-any.whl.metadata (5.2 kB)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from lightning-utilities>=0.8.0->torchmetrics) (71.0.4)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from lightning-utilities>=0.8.0->torchmetrics) (4.12.2)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->torchmetrics) (3.15.4)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->torchmetrics) (1.13.1)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->torchmetrics) (3.3)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->torchmetrics) (3.1.4)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->torchmetrics) (2024.6.1)
    Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch>=1.10.0->torchmetrics)
      Using cached nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cuda-runtime-cu12==12.1.105 (from torch>=1.10.0->torchmetrics)
      Using cached nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cuda-cupti-cu12==12.1.105 (from torch>=1.10.0->torchmetrics)
      Using cached nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cudnn-cu12==8.9.2.26 (from torch>=1.10.0->torchmetrics)
      Using cached nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cublas-cu12==12.1.3.1 (from torch>=1.10.0->torchmetrics)
      Using cached nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cufft-cu12==11.0.2.54 (from torch>=1.10.0->torchmetrics)
      Using cached nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-curand-cu12==10.3.2.106 (from torch>=1.10.0->torchmetrics)
      Using cached nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cusolver-cu12==11.4.5.107 (from torch>=1.10.0->torchmetrics)
      Using cached nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cusparse-cu12==12.1.0.106 (from torch>=1.10.0->torchmetrics)
      Using cached nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-nccl-cu12==2.20.5 (from torch>=1.10.0->torchmetrics)
      Using cached nvidia_nccl_cu12-2.20.5-py3-none-manylinux2014_x86_64.whl.metadata (1.8 kB)
    Collecting nvidia-nvtx-cu12==12.1.105 (from torch>=1.10.0->torchmetrics)
      Using cached nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.7 kB)
    Requirement already satisfied: triton==2.3.1 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->torchmetrics) (2.3.1)
    Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch>=1.10.0->torchmetrics)
      Using cached nvidia_nvjitlink_cu12-12.6.20-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.10.0->torchmetrics) (2.1.5)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.10.0->torchmetrics) (1.3.0)
    Downloading torchmetrics-1.4.1-py3-none-any.whl (866 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m866.2/866.2 kB[0m [31m4.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading lightning_utilities-0.11.6-py3-none-any.whl (26 kB)
    Using cached nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
    Using cached nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
    Using cached nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
    Using cached nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
    Using cached nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
    Using cached nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
    Using cached nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
    Using cached nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
    Using cached nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
    Using cached nvidia_nccl_cu12-2.20.5-py3-none-manylinux2014_x86_64.whl (176.2 MB)
    Using cached nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
    Using cached nvidia_nvjitlink_cu12-12.6.20-py3-none-manylinux2014_x86_64.whl (19.7 MB)
    Installing collected packages: nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, lightning-utilities, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, torchmetrics
    Successfully installed lightning-utilities-0.11.6 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.20.5 nvidia-nvjitlink-cu12-12.6.20 nvidia-nvtx-cu12-12.1.105 torchmetrics-1.4.1
    


```python
# Install the Kaggle library
! pip install kaggle

# Make a directory named ".kaggle"
! mkdir ~/.kaggle

# Copy the "kaggle.json" into this new directory
! cp kaggle.json ~/.kaggle/

# Allocate the required permission for this file
! chmod 600 ~/.kaggle/kaggle.json

! kaggle competitions download -c digit-recognizer

! unzip digit-recognizer.zip
```

    Requirement already satisfied: kaggle in /usr/local/lib/python3.10/dist-packages (1.6.17)
    Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.10/dist-packages (from kaggle) (1.16.0)
    Requirement already satisfied: certifi>=2023.7.22 in /usr/local/lib/python3.10/dist-packages (from kaggle) (2024.7.4)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.8.2)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.32.3)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from kaggle) (4.66.5)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.10/dist-packages (from kaggle) (8.0.4)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.10/dist-packages (from kaggle) (2.0.7)
    Requirement already satisfied: bleach in /usr/local/lib/python3.10/dist-packages (from kaggle) (6.1.0)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from bleach->kaggle) (0.5.1)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.10/dist-packages (from python-slugify->kaggle) (1.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->kaggle) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->kaggle) (3.7)
    Downloading digit-recognizer.zip to /content
      0% 0.00/15.3M [00:00<?, ?B/s]
    100% 15.3M/15.3M [00:00<00:00, 164MB/s]
    Archive:  digit-recognizer.zip
      inflating: sample_submission.csv   
      inflating: test.csv                
      inflating: train.csv               
    

# 2. Load Training and Testing Data


```python
# read 'train.csv' and 'test.csv' which are comma-separated values (csv) file into DataFrame.
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"The training data has {train_df.shape[0]} rows and {train_df.shape[1]} columns.")
print(f"The testing data has {test_df.shape[0]} rows and {test_df.shape[1]} columns.")
```

    The training data has 42000 rows and 785 columns.
    The testing data has 28000 rows and 784 columns.
    


```python
train_df.head()
```





  <div id="df-31f331f3-dbdb-42ea-b672-eddeb8336390" class="colab-df-container">
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
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-31f331f3-dbdb-42ea-b672-eddeb8336390')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-31f331f3-dbdb-42ea-b672-eddeb8336390 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-31f331f3-dbdb-42ea-b672-eddeb8336390');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-07d14d0b-f1ad-4b2a-8951-7568f96a4e5b">
  <button class="colab-df-quickchart" onclick="quickchart('df-07d14d0b-f1ad-4b2a-8951-7568f96a4e5b')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-07d14d0b-f1ad-4b2a-8951-7568f96a4e5b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
test_df.head()
```





  <div id="df-7b4f074f-7945-458c-82c5-c0d38220d441" class="colab-df-container">
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
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 784 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7b4f074f-7945-458c-82c5-c0d38220d441')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7b4f074f-7945-458c-82c5-c0d38220d441 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7b4f074f-7945-458c-82c5-c0d38220d441');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-938a824f-dc9a-4acd-b83a-fea8a10842f8">
  <button class="colab-df-quickchart" onclick="quickchart('df-938a824f-dc9a-4acd-b83a-fea8a10842f8')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-938a824f-dc9a-4acd-b83a-fea8a10842f8 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
train_df.describe()
```





  <div id="df-dd40afa3-2ee1-4e9d-b1ad-4cddb0c3a4ad" class="colab-df-container">
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
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>42000.000000</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>...</td>
      <td>42000.000000</td>
      <td>42000.000000</td>
      <td>42000.000000</td>
      <td>42000.00000</td>
      <td>42000.000000</td>
      <td>42000.000000</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
      <td>42000.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.456643</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.219286</td>
      <td>0.117095</td>
      <td>0.059024</td>
      <td>0.02019</td>
      <td>0.017238</td>
      <td>0.002857</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.887730</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>6.312890</td>
      <td>4.633819</td>
      <td>3.274488</td>
      <td>1.75987</td>
      <td>1.894498</td>
      <td>0.414264</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>254.000000</td>
      <td>254.000000</td>
      <td>253.000000</td>
      <td>253.00000</td>
      <td>254.000000</td>
      <td>62.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 785 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-dd40afa3-2ee1-4e9d-b1ad-4cddb0c3a4ad')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-dd40afa3-2ee1-4e9d-b1ad-4cddb0c3a4ad button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-dd40afa3-2ee1-4e9d-b1ad-4cddb0c3a4ad');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-6a966a34-2ae5-497c-aec9-ba728cf81973">
  <button class="colab-df-quickchart" onclick="quickchart('df-6a966a34-2ae5-497c-aec9-ba728cf81973')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-6a966a34-2ae5-497c-aec9-ba728cf81973 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




The training and test datasets contain 42,000 and 28,000 grayscale images of hand-drawn digits from 0 to 9, respectively.

The training dataset includes 785 columns, one more than the test dataset. The first column, labeled `label`, indicates the digit written by the user.

Each image consists of 784 pixels (28 pixels in height and 28 pixels in width), corresponding to the remaining columns. Pixel values range from 0 to 255, representing the brightness of each pixel, with higher values indicating darker pixels.

Let's examine the number of images in each digit class.


```python
train_df['label'].value_counts(normalize=True)
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
      <th>proportion</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.111524</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.104786</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.103595</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.099714</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.099452</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.098500</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.098381</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.096952</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.096738</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.090357</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



The images appear to be fairly evenly distributed across the digit classes. Given that there are 10 different digit classes (0 through 9), this is a multi-class classification problem.

# 3. Transform Data and Prepare DataLoader

Before using the data in PyTorch, we need to convert it into tensors. A tensor is a mathematical object that generalizes the concepts of scalars, vectors, and matrices to higher dimensions. In machine learning and deep learning, tensors represent data in a structured format that can be efficiently processed by algorithms, particularly neural networks.


After converting the data into tensors, we'll use `torch.utils.data.DataLoader`, which combines a dataset with a sampler and provides an iterable for easy access to the dataset.




```python
# convert dataframe to numpy array
train_arr = train_df.to_numpy()

# separate 'train_arr' into X (pixels) and y (label)
X = train_arr[:, 1:].reshape((train_df.shape[0], 28, 28, 1)).astype(np.uint8) # NCHW (Number of Images, Color Channels, Height, Width)
y = train_arr[:, 0]

# one-hot encode y
enc = OneHotEncoder()
y = enc.fit_transform(y.reshape(-1, 1)).toarray()
```

Using [`ToTensor()`](https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor), a PIL Image or a `numpy.ndarray` with dimensions (H x W x C) in the range [0, 255] is transformed into a `torch.FloatTensor` with shape (C x H x W) and values scaled to the range [0.0, 1.0]. This applies when the PIL Image is in one of the supported modes (such as L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the `numpy.ndarray` has a data type of `np.uint8`. To prepare our data accordingly, we converted `train_df` into a Numpy array with the shape (42000, 28, 28, 1) and a data type of `np.uint8`.


```python
# create 'data_transform' that converts X into tensors
data_transform = transforms.ToTensor()
```

Now, we'll plot 12 randomly selected images of handwritten digits after they've been transformed into tensors.


```python
torch.manual_seed(42)

# set figure size
fig = plt.figure(figsize=(12, 7))

# we want to plot 12 plots
rows, cols = 4, 3

for i in range(1, rows*cols+1):
  # choose a random index
  random_idx = torch.randint(0, len(X), size=[1]).item()

  # convert the chosen image into a tensor
  img = data_transform(X[random_idx])

  # store the image's label
  label = y[random_idx]

  # plot the image
  fig.add_subplot(rows, cols, i)
  plt.imshow(img.squeeze(), cmap='gray')
  plt.title(np.where(label == 1)[0][0])
  plt.axis(False);
```



![output_17_0](https://github.com/user-attachments/assets/33cd3848-0fd9-4fe4-9dd8-5bd8d6677ce8)



Now, we're ready to set up the DataLoader, which allows us to break down a large dataset into a Python iterable of smaller mini-batches.


```python
# one-hot encode the label
label_oh = pd.get_dummies(train_df['label'], prefix='label', dtype=int)

# drop the label
train_df.drop(['label'], axis=1, inplace=True)

# concatenate the one-hot encoded label and the features
train_df = pd.concat([label_oh, train_df], axis=1)
```


```python
# for training: first 28000 images from the training data
class DigitRecognizerTrainDataset(torch.utils.data.Dataset):
    def __init__(self):
      self.dataset = train_df[:28000].to_numpy()

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        features, label = sample[10:], sample[:10]
        features = features.reshape((28, 28, 1)).astype(np.uint8)

        transform = transforms.ToTensor()
        return transform(features), torch.tensor(label)

    def __len__(self):
        return len(self.dataset)

# for testing: the remaining images
class DigitRecognizerTestDataset(torch.utils.data.Dataset):
    def __init__(self):
      self.dataset = train_df[28000:].to_numpy()

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        features, label = sample[10:], sample[:10]
        features = features.reshape((28, 28, 1)).astype(np.uint8)

        transform = transforms.ToTensor()
        return transform(features), torch.tensor(label)

    def __len__(self):
        return len(self.dataset)
```


```python
train_set = DigitRecognizerTrainDataset()
test_set = DigitRecognizerTestDataset()
```


```python
# set up batch size
batch_size = 32

# turn datasets into batches
train_dataloader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True)

test_dataloader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False)

print(f"There are {len(train_dataloader)} batches of size {batch_size} in train dataloader.")
print(f"There are {len(test_dataloader)} batches of size {batch_size} in test dataloader.")
```

    There are 875 batches of size 32 in train dataloader.
    There are 438 batches of size 32 in test dataloader.
    


```python
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape, train_labels_batch.shape
```




    (torch.Size([32, 1, 28, 28]), torch.Size([32, 10]))



# 4. Modeling

With the training and test data loaders prepared, we can now begin building the model by subclassing `nn.Module`, the base class for all neural network modules.


```python
class DigitRecognizerModel(nn.Module):
  def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
    super().__init__()
    self.layer_stack = nn.Sequential(
        # compresses the dimensions of a tensor into a single vector.
        # [C, H, W] -> [C, H*W]
        nn.Flatten(),
        nn.Linear(in_features=input_shape, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=output_shape),
        nn.Softmax(dim=1)
    )
  def forward(self, x):
    return self.layer_stack(x)
```

Let's instantiate a model using `DigitRecognizerModel`, but first, we need to set the following parameters.

- `input_shape` represents the number of features provided to the model. For this example, we'll set `input_shape` to 784, which corresponds to 28 pixels in height by 28 pixels in width.

- `hidden_units` specifies the number of neurons in the hidden layer(s). It typically ranges between 10 and 512. You can experiment with different values within this range to find the optimal performance on the test data. In this case, we'll set it to 100.

- `output_shape` is set to 10 since we are classifying 10 digits (from 0 to 9).

We can add more hidden layers to create more complex models, and the number of `hidden_units` can vary for each hidden layer.

Let's go ahead and create an instance of `DigitRecognizerModel`.


```python
torch.manual_seed(42)

model = DigitRecognizerModel(input_shape=28*28,
                             hidden_units=100,
                             output_shape=y.shape[1])
model.to('cpu')
```




    DigitRecognizerModel(
      (layer_stack): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Linear(in_features=784, out_features=100, bias=True)
        (2): ReLU()
        (3): Linear(in_features=100, out_features=10, bias=True)
        (4): Softmax(dim=1)
      )
    )



Before training the model, we need to configure the appropriate loss function, optimization method, and evaluation metrics for our multi-class classification problem.

For evaluation, we use `torchmetrics.Accuracy`, which calculates the fraction of correct predictions made by the model. If the predictions are in floating-point format, `torch.argmax` is applied along the label classes to convert probabilities or logits into integer tensor values.

For the loss function, we'll use cross-entropy loss, and the Stochastic Gradient Descent (SGD) algorithm will be employed as the optimizer.


```python
# set up accuracy function, loss function and optimizer
acc_fn = torchmetrics.Accuracy(task='multiclass', num_classes=10)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
```

We’re now set to begin training the model!


```python
# set the seed
torch.manual_seed(42)

# set the number of epochs
epochs = 100

train_loss_list = []
test_loss_list = []
test_acc_list = []

# create training and testing loop
for epoch in range(epochs):

  # training
  train_loss = 0
  for batch, (X, y) in enumerate(train_dataloader):
    model.train()
    # forward pass
    y_pred = model(X)
    # compute loss (per batch)
    loss = loss_fn(y_pred, y.type(torch.FloatTensor))
    train_loss += loss
    # optimizer zero grad
    optimizer.zero_grad()
    # loss backward
    loss.backward()
    # optimizer step
    optimizer.step()

  train_loss /= len(train_dataloader)
  train_loss_list.append(train_loss.item())

  # testing
  test_loss, test_acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in test_dataloader:
      # forward pass
      test_pred = model(X)
      # compute loss
      test_loss += loss_fn(test_pred, y.type(torch.FloatTensor))
      # compute accuracy
      test_acc += acc_fn(test_pred, torch.tensor(np.where(y == 1)[1]))

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    test_loss_list.append(test_loss.item())
    test_acc_list.append(test_acc.item())

  if (epoch % 10) == 0:
    print(f"Epoch: {epoch}\n------")
    print(f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {100*test_acc:.2f}%\n")
```

    Epoch: 0
    ------
    Train loss: 1.9075 | Test loss: 1.6826, Test acc: 81.91%
    
    Epoch: 10
    ------
    Train loss: 1.5308 | Test loss: 1.5373, Test acc: 93.19%
    
    Epoch: 20
    ------
    Train loss: 1.5087 | Test loss: 1.5200, Test acc: 94.68%
    
    Epoch: 30
    ------
    Train loss: 1.4967 | Test loss: 1.5117, Test acc: 95.28%
    
    Epoch: 40
    ------
    Train loss: 1.4900 | Test loss: 1.5076, Test acc: 95.63%
    
    Epoch: 50
    ------
    Train loss: 1.4855 | Test loss: 1.5059, Test acc: 95.74%
    
    Epoch: 60
    ------
    Train loss: 1.4823 | Test loss: 1.5036, Test acc: 95.88%
    
    Epoch: 70
    ------
    Train loss: 1.4803 | Test loss: 1.5020, Test acc: 96.18%
    
    Epoch: 80
    ------
    Train loss: 1.4788 | Test loss: 1.5010, Test acc: 96.20%
    
    Epoch: 90
    ------
    Train loss: 1.4775 | Test loss: 1.5003, Test acc: 96.22%
    
    

After 100 epochs, the model reaches an accuracy of 96.22%. As we can see in the plot below, the loss on both the training and test datasets steadily decreased as the model was trained.


```python
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.legend()
plt.show()
```


    
![output_33_0](https://github.com/user-attachments/assets/8f04e2d4-34e1-44fb-9cb4-20d10d377d20)
 


We also create a confusion matrix to evaluate the accuracy of our model's predictions against the actual labels.


```python
test_cm_dataloader = DataLoader(test_set,
                                batch_size=1,
                                shuffle=False)
```


```python
# make predictions across all test data using the trained model
model.eval()

y_preds = []
y_true = []

with torch.inference_mode():
  for X, y in test_cm_dataloader:
    y_true.append(np.where(y[0] == 1)[0][0])
    # forward pass
    y_pred_logits = model(X)
    # logits -> probability -> label
    y_pred_labels = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
    # append the labels to y_preds
    y_preds.append(y_pred_labels.item())
```


```python
# Setup confusion matrix
confmat = ConfusionMatrix(task="multiclass", num_classes=10)
confmat_tensor = confmat(preds=torch.tensor(y_preds),
                         target=torch.tensor(y_true))

# Plot the confusion matrix
fix, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=np.arange(0, 10),
    figsize=(10, 7)
)
```


    
![output_37_0](https://github.com/user-attachments/assets/27a223c4-694d-4a61-8c78-635e78da52e8)



# 5. Making Predictions on the Data for Submission

The final step is to make predictions on the test dataset and submit them to evaluate their accuracy.


```python
class DigitRecognizerSubmissionDataset(torch.utils.data.Dataset):
    def __init__(self):
      self.dataset = test_df.to_numpy()

    def __getitem__(self, idx):
        features = self.dataset[idx]
        features = features.reshape((28, 28, 1)).astype(np.uint8)

        transform = transforms.ToTensor()
        return transform(features)

    def __len__(self):
        return len(self.dataset)
```


```python
submission_set = DigitRecognizerSubmissionDataset()
```


```python
submission_dataloader = DataLoader(submission_set,
                                   batch_size=len(submission_set),
                                   shuffle=False)
```


```python
model.eval()
with torch.inference_mode():
  for X in submission_dataloader:
    label_logits = model(X)
```


```python
label_probs = torch.softmax(label_logits, dim=1)
Label = torch.argmax(label_probs, dim=1)
```


```python
submission = pd.DataFrame(data=Label.numpy(), index=np.arange(1, test_df.shape[0]+1),columns=['Label'])
submission.index.name = 'ImageId'
```


```python
submission.to_csv('submission_digit_recognizer.csv')
```
