# Exploratory Data Analysis

### Problem statement
Perform EDA on the given dataset to predict possibility of 'stroke' based on the feature vectors given in the dataset

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Let us import the data set called stroke into python environment

df= pd.read_csv("/kaggle/input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv")
df

#looking at the top 5 rows
print('Below are the first 5 rows of the dataset')
df.head()

#looking at bottom 5 rows
print('Below are the bottom 5 rows of the dataset')
df.tail()

df.columns

#checking number of rows and columns
df.shape

* **id:** Unique identifier
* **gender:** Gender of the patient (Male, Female, Other)
* **age:** Age of the patient
* **hypertension:** **0** if the patient doesn't have hypertension, **1** if the patient has hypertension
* **heart_disease:** **0** if the patient doesn't have any heart diseases, **1** if the patient has a heart disease
* **ever_married:** **Yes** if the patient is married, **No** if the patient is not married 
* **work_type:** Profession of the patient (children, Govt_job, Never_worked, Private, Self-employed)
* **Residence_type:** Residence category of the patient (Rural, Urban)
* **avg_glucose_level:** Average glucose level in blood of the patient
* **bmi:** Body Mass Index of the patient
* **smoking_status:** Smoking status of the patient (formerly smoked, never smoked, smokes, Unknown). **Unknown** in **smoking_status** means that the information is unavailable for this patient
* **stroke:** **1** if the patient had a stroke or **0** if not




#Creating a list of of all the ccolumns in the dataset
columns_list = list(df.columns)

#Getting unique values in each column
for i in columns_list:
    print('Number of unique values in column',i, 'is', df[i].nunique())



By the initial glance, we can understand that id will not play any role as it is just a unique number possibly given to each patient.
Hence, we can drop the id column.

#Dropping ID column and storing the 'id' dropped dataframe in df_v1
df_v1= df.drop('id', axis=1)

Updating list of columns after dropping id column

columns_list = list(df_v1.columns)

df_v1.head(100)

#Checking for statistical summary
df_v1.describe()

df_v1.describe(include=object)

df_v1.info()

**Insights**

1. Count of non-null values in column 'bmi' is 4909 which is less than the count in the other columns.
2. There is no data type conversion required.
3. Label encoding would be required for columns, 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'

#### 1. Data Cleaning

Raw data collected from domains would be incorrect and inoperable and possible reasons could be mistyping, corruption, duplication, missing valuess and so on. And the basic data cleaning has to exercised before exercising any further steps of data pre-processing.

Further Data pre-processing techniques like Feature selection, Data transforms, feature engineering and dimensionality reduction are necessitated by Exploratory Data Analysis.

I follow a WHH approach to achieve data cleaning.
* **W**- What to find?
* **H**- How to find?
* **H**- How to treat?


**1. What to find? : Missing Values** 

Missing data refers to cells of a tabular dataset in which the data is not stored. These cells will not contain any real values.

**How to find missing values?**

#Checking for null values in each column
df_v1.isnull().sum()

df_v1.isnull().sum().to_frame().T

**How to treat missing values?**

Method 1: Dropping rows or columns.


#Dropping rows that contain missing values and storing in df_drop
df_drop = df_v1.dropna()

#validating drop action
df_drop.isna().sum()

df_drop.shape

Method 2: Imputation - filling missing values with mean or median or mode of the resspective column.


#method2: Imputation - filling missing values with mean or median of the resspective column

bmi_mean = round(df_v1['bmi'].mean(), 1)
#bmi_mean = (df_v1['bmi'].mean())

bmi_med = df_v1['bmi'].median()

print(bmi_mean)
print(bmi_med)

#Filling missing values with bmi_mean and storing new dataframe in df_imp
df_imp = df_v1.fillna(bmi_mean, axis=1)

df_imp

#validating imputation action
df_imp.isna().sum()

df.shape

**NOTE: We know that BMI - Body mass Index plays a major role in determining the possibility of stroke. Hereon, let's use the dataframe df_imp for exploration.**

**2. What to find? : Duplicate Values** 

Values are considered to be duplicate when the entire row is found to be identical with any ther row and the same goes for columns as well.
Sometimes the datapoint duplication will be defined by only a few set of columns as well.

**How to find duplicated values?**

#duplicate rows
df_imp.duplicated().sum()

There are no duplicate values in the dataset.

However, it is not so straight forward to find out duplicate or redundant columns. A clear understanding of the dataset and the respective columns is required to determine column duplication. It is the pattern to be determined by a deep analysis.

**How to treat duplicated values?**

***df.drop_duplicates()*** is the function used to drop duplicate values. By default the function retains the first value.

**3. What to find? : Outliers** 

An outlier is an observation that lies at an abnormal distance from other values in a random sample from a population.

**How to find outliers?**

Method 1: Box and whisker plots
![image.png](attachment:image.png)

Method 2: By the definition of how many standard deviations away from the mean.

***mean +/- n * Standard deviation***

where, **n** is the number of standard deviations which will have to be defined by a domain expert.


df_imp.columns

The numerical features available in the dataset are 'age', 'hypertension', 'heart_disease', 'avg_glucose_level' and 'bmi'. However, the features 'hypertension', 'heart_disease' are necessarily encoded categorical features contaiining only 2 unique values '0' and '1'.

Hence, we will check for outliers only in columns 'age', 'avg_glucose_level' and 'bmi'.

Let me just put a box and whisker plot(Method 1) to understand outliers in the given dataset as we do not have any domain expert to define the 'n'(Method 2). 

#Box and Whisker plot for column - 'age'
sns.boxplot(data=df_imp, x='age')

**Inference:**
There are no outliers in the age column.

#just having a look at the distribution of the age column
plt.figure(figsize=(5,5))
sns.histplot(df_imp['age'], color='tomato', label='BMI',kde=True)
plt.legend()
plt.grid()

#Box and Whisker plot for column - 'avg_glucose_level'

sns.boxplot(data=df_imp, x='avg_glucose_level')

There are outliers in the Average glucose level

#Box and Whisker plot for column - 'bmi'
sns.boxplot(data=df_imp, x='bmi')

##### Let us analyse the target vector-'stroke' and see how many class labels are available

df_imp['stroke'].value_counts()

df_grp = pd.DataFrame(df_imp.groupby('stroke')['stroke'].count())
df_grp
colors = ['darkcyan','turquoise']
labels=['No Stroke', 'Stroke']
plt.pie([4861,249], labels=labels,autopct='%.3f%%', radius=1.5, colors=colors, shadow=True, explode=[0,0.1])
plt.show()
fig = plt.gcf()
fig.set_facecolor('black')

From the Pie chart for the target vector 'stroke', we can observe that the percentage of those who did not have stroke is higher in coomparison to the percentage of those who had stroke.
i.e., Percentage of occurence of 'stroke' << Percentage of occurence of 'no stroke'

And this infers that the data set set is **largely imbalanced**.

Checking the histogram of each column

df_imp.columns

plt.figure(figsize=(10,20))
for i in range(len(columns_list)):
    plt.subplot(5,3,i+1)
    plt.title(columns_list[i])
    plt.xticks(rotation=45)
    plt.hist(df_imp[columns_list[i]])
    
plt.tight_layout()


1. Looking at the histogram of gender column looks like there are 3 different unique values. Count of 'Other' in gender column seems to be too low. 
2. Avergae glucose level is right skewed
3. BMI column seems to be leptokurtic

#Unique values in column gender
df_imp['gender'].value_counts()

#checking the row where 'gender' column contaiins 'other'
df_imp[df_imp['gender']=='Other']

#dropping row number-3116(where gender=Other)
df_imp = df_imp.drop(3116)

#### 1. Univariate Analysis

- Analysing one variable/feature/column at a time. Uni-One.
- Univariate analysis can be descriptive or inferential. 
- We can use any univariate plots to demonstrate univariate analysis like histogram, PDF, CDF, BOX plot, violin plot, 1D-scatterplot...and so on.

#Understanding the PDF of 'age', 'avg_glucose_level', 'bmi'
num_list = ['age', 'avg_glucose_level', 'bmi']
plt.figure(figsize=(10,4))
for i in range(len(num_list)):
    plt.subplot(1,3,i+1)
    sns.histplot(data=df_imp,x=num_list[i], hue='stroke', kde=True, stat="probability", common_norm=False)
    plt.title('Stroke vs No Stroke by {}'.format(num_list[i]))
plt.tight_layout()

Inference:
1. Age: When age is less than 40, chances of stroke are slim; When age is more than 50 chances of atroke is high;At the age of 70-80, chances of stroke is very high.
2. Glucose level: Chances of stroke is high if glucose level is more than 150.
        

df_imp.columns

#putting all object datatype columns in a list
obj_list = ['hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status']

#plotting the count plot for each column and looking at the number of counts for strke and no stroke
plt.figure(figsize=(7,30))
for i in range(len(obj_list)):
    plt.subplot(6,1,i+1)
    sns.countplot(data=df_imp, x=obj_list[i], hue='stroke')
plt.tight_layout()

* **Hypertension countplot**:  

#### 2. Bivariate Analysis

- Analysing two variable/feature/column at a time. Bi-Two.
- Bivariate analysis can be descriptive or inferential. 
- We can use any bivariate plots to demonstrate univariate analysis like scatter plot, line plot, Box plot(SNS), Violin Plot(SNS)...etc

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_imp, x="age", y="bmi", hue='stroke')
plt.show()

plt.figure(figsize=(60,40))
sns.pairplot(df_imp, hue='stroke', height=7, palette='hsv')
plt.savefig('D:\\Dataset\\pair_plot.png', format='png')

#### 3. Multivariate Analysis

- Analysing more than 2 variables/features/columns at a time.
- Bivariate analysis can be descriptive or inferential. 
- We can use any multivariate plots to demonstrate univariate analysis like 3D scatter plot, Contour plots, heatmaps.

import plotly.express as px
fig = px.scatter_3d(df_imp, x='age', y='bmi', z='avg_glucose_level',
              color='stroke',color_discrete_sequence=px.colors.qualitative.Bold, width=1200, height=1200)
fig.show()
