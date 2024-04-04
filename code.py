# import the necessary libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
import ast

# import the dataset
df = pd.read_csv('adult.csv')
# print the head of the df
print(df.head(10))
# from here we can see it is supervised machine learning because income parameter is binary by nature

# print shape of dataset
print(df.shape)

# check the data types
print(df.dtypes)

# check how many datasets are NULL
print(df.isnull().sum())

# based on the head set ? is identified. Need to check the entire data set

# check the values of all the unique features
print(df.nunique())

# check the transpose of the data
print(df.describe().T)

# VALUE COUNT FUNCTION for workclass
print(df['workclass'].value_counts())

# based on the value count check the ? has 2799 variables so it cannot be dropped it will affect the ML

# LETS CHECK THE COLUMNS INDEX
print(df.columns)

# VALUE COUNT FUNCTION for occupation
print(df['occupation'].value_counts())

# based on occupation we have ? a total of 2809 characters, Prof-specialty is the highest with 6172
# fill the ? with the mode

# VALUE COUNT FUNCTION for native-country
print(df['native-country'].value_counts())
# check for native country is 857 ?, United states is the highest with 43832

# VALUE COUNT FUNCTION for marital-status
print(df['marital-status'].value_counts())

# The dataset is clean most are married-civ-spouse 22379

# VALUE COUNT FUNCTION for race
print(df['race'].value_counts())

# data is clean with white being the highest race of 41762

# VALUE COUNT FUNCTION for gender
print(df['gender'].value_counts())

# Binary value for male and female with more males then females 32650

# VALUE COUNT FUNCTION for income
print(df['income'].value_counts())

# binary dataset with more than 37155 earning <= 50,000

# sns check income between the gender
# sns for jyputer notebook
#sns.countplot(df['income'], palette = 'coolwarm', hue = 'gender', data = df);

sns.countplot(df['income'])
plt.show()
#sns.countplot(df['income'], ['gender'])
#plt.show()
#df_long = df.melt(id_vars=['income'], var_name='gender', value_name='count')
#sns.histplot(x='income', hue='gender', data=df_long)  # Use long-form data
#plt.show()
# sns check income between the gender
# sns for jyputer notebook
#sns.countplot(df['income'], palette = 'coolwarm', hue = 'gender', data = df);
#sns.histplot(df['income'])
#plt.show()
#sns.countplot(df['income'], ['gender'])
#plt.show()
# Assuming your DataFrame is called 'df'
#sns.countplot(df['income', hue='gender'], data=df, palette='RdBu', size=6, linewidth=2])
#plt.show()
#df_long = df.melt(id_vars=['income'], var_name='race', value_name='count')
#sns.histplot(x='income', hue='race', data=df_long)  # Use long-form data
#sns.countplot(df['income'], [data = df]);
#plt.show()

# replace values with mode
df['workclass'] = df['workclass'].replace('?', 'Private')
df['occupation'] = df['occupation'].replace('?', 'Prof-specialty')
df['native-country'] = df['native-country'].replace('?', 'United-States')

# the check
print(df.head(10))

print(df['native-country'].value_counts())

# Freature engineer education cat

print(df['education'].value_counts())
# replace all the lower edu as school
df['education'] = df['education'].replace(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th'], 'school')
print(df['education'].value_counts())

df['education'] = df['education'].replace('HS-grad', 'high school')
df['education'] = df['education'].replace(['Assoc-acdm', 'Assoc-voc', 'Prof-school', 'Some-college'], 'higher')
df['education'] = df['education'].replace('Bachelors', 'undergrad')
df['education'] = df['education'].replace('Masters', 'grad')
df['education'] = df['education'].replace('Doctorate','doc')
print(df['education'].value_counts())


# marital status
print(df['marital-status'].value_counts())
df['marital-status'] = df['marital-status'].replace(['Married-civ-spouse', 'Married-AF-spouse',], 'married')
df['marital-status'] = df['marital-status'].replace('Never-married','not-married')
df['marital-status'] = df['marital-status'].replace(['Divorced', 'Separated', 'Widowed', 'Married-spouse-absent'], 'other')
print(df['marital-status'].value_counts())

#income
df['income'] = df['income'].replace('<=50K', 0)
df['income'] = df['income'].replace('>50K', 1)

print(df['income'].value_counts())

print(df.head())

#df.corr()

sns.heatmap(df.corr(), annot=True);

