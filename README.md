# Predicting Titanic Survival Rates with Random Forest Classifier

This repository contains a complete implementation of a machine learning model to predict survival rates from the Titanic dataset using a Random Forest Classifier. The project includes data loading, preprocessing, exploratory data analysis, model training, evaluation, and submission.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Submission](#submission)
- [Usage](#usage)

## Introduction

The Titanic dataset is a well-known dataset used in machine learning for predicting whether a passenger survived based on various features such as age, gender, and ticket class. This implementation utilizes a Random Forest Classifier to make predictions.

## Requirements

To run the code, ensure you have the following Python packages installed:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `fastai`
- `graphviz`

You can install these packages using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn fastai graphviz
```

## Data Preparation

The data preparation process involves loading the dataset and preprocessing it for analysis. The following steps are performed:

1. **Load Data**: The training and test datasets are loaded from CSV files.
2. **Handle Missing Values**: Missing values in the 'Fare' column are filled with 0, and other missing values are filled with the mode of their respective columns.
3. **Feature Engineering**: A logarithmic transformation is applied to the 'Fare' column for normalization.

```python
import os
import pandas as pd
import numpy as np

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle:
    path = Path('../input/titanic')
else:
    import zipfile, kaggle
    path = Path('titanic')
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)

df = pd.read_csv(path/'train.csv')
tst_df = pd.read_csv(path/'test.csv')
modes = df.mode().iloc[0]

def proc_data(df):
    df['Fare'] = df.Fare.fillna(0)
    df.fillna(modes, inplace=True)
    df['LogFare'] = np.log1p(df['Fare'])
    df['Embarked'] = pd.Categorical(df.Embarked)
    df['Sex'] = pd.Categorical(df.Sex)

proc_data(df)
proc_data(tst_df)
```

## Exploratory Data Analysis

In this section, we analyze the data to understand survival rates based on different features.

1. **Visualizing Survival Rates**: Bar plots and count plots are created to visualize survival rates by gender.
2. **Feature Analysis**: Boxen plots and KDE plots are used to analyze the distribution of 'LogFare' based on survival.

```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(11, 5))
sns.barplot(data=df, y='Survived', x="Sex", ax=axs[0]).set(title="Survival rate")
sns.countplot(data=df, x="Sex", ax=axs[1]).set(title="Histogram")
```

## Model Training

The model training process involves splitting the dataset into training and validation sets, transforming categorical variables into numerical codes, and fitting a Random Forest Classifier.

1. **Data Splitting**: The training data is split into training and validation sets.
2. **Model Fitting**: A Random Forest Classifier is trained on the processed data.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

random.seed(42)
trn_df, val_df = train_test_split(df, test_size=0.75)

cats = ["Sex", "Embarked"]
conts = ['Age', 'SibSp', 'Parch', 'LogFare', "Pclass"]
dep = "Survived"

trn_df[cats] = trn_df[cats].apply(lambda x: x.cat.codes)
val_df[cats] = val_df[cats].apply(lambda x: x.cat.codes)

def xs_y(df):
    xs = df[cats + conts].copy()
    return xs, df[dep]

trn_xs, trn_y = xs_y(trn_df)
val_xs, val_y = xs_y(val_df)

rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5)
rf.fit(trn_xs, trn_y)
```

## Model Evaluation

The model's performance is evaluated using mean absolute error on the validation set.

```python
from sklearn.metrics import mean_absolute_error

predictions = rf.predict(val_xs)
error = mean_absolute_error(val_y, predictions)
print(f'Mean Absolute Error: {error}')
```

## Submission

Finally, predictions are made on the test set and saved in a CSV file for submission.

```python
tst_df[cats] = tst_df[cats].apply(lambda x: x.cat.codes)
tst_xs,_ = xs_y(tst_df)

def subm(preds, suff):
    tst_df['Survived'] = preds
    sub_df = tst_df[['PassengerId', 'Survived']]
    sub_df.to_csv(f'sub-{suff}.csv', index=False)

subm(rf.predict(tst_xs), 'rf')
```

## Usage

To use this code:

1. Clone this repository.
2. Install the required packages.
3. Run the Jupyter Notebook or Python script to execute each section sequentially.
4. Review the results and adjust parameters as needed for improved accuracy.

This README outlines how to use the code for predicting Titanic survival rates with a Random Forest Classifier. You can adjust parameters or explore additional features to improve model performance as needed.
