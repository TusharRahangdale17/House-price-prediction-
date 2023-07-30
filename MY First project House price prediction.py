#!/usr/bin/env python
# coding: utf-8

# # House price prediction Regression project 

# # Project Goal

# # Load libraries

# In[21]:


# load libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
from scipy.stats import mstats
import scipy.stats as stats
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler


# # load Dataset

# In[51]:


train_data_path = r"C:\Users\TUSHAR\Desktop\House_price_prediction\data\train.csv"

data= pd.read_csv(train_data_path)

print("shape of df_train: ", df_train.shape)


# In[52]:


# this is used for view all columns
pd.set_option("display.max_columns",None)
data.head


# In[53]:


# Get the shape of the data (number of rows,number of columns)
print(data.shape)


# In[55]:


# Get information about the columns and data types 
data.info()


# In[56]:


# Get summary statistics of the numerical columns
data.describe()


# In[57]:


# Assign Categorical and numeric columns based on data types
categorical_cols = data.select_dtypes(include= 'object')
numeric_cols = data.select_dtypes(include=['int','float']).columns


# In[58]:


# Check the unique values in each categorial column
for col in categorical_cols:
    print(data[col].unique())


# # Finding & handling Missing Data

# In[59]:


# Check for mising value in the dataset 
#pd. set_option('dispaly.max_rows', None) # this is so we can view all row
data.isnull().sum()


# In[60]:


# handle missing value bsed on the context of the data
data['LotFrontage'].fillna(data['LotFrontage'].mean(),inplace= True)
data['MasVnrArea'].fillna(0,inplace= True)
data['GarageYrBlt'].fillna(data['GarageYrBlt'].median(),inplace= True)
data.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[61]:


object_columns_with_missing_values = data.select_dtypes(include='object').columns[data.select_dtypes(include='object').isnull().any()]
object_columns_with_missing_values


# In[62]:


# Iterate over object columns with missing values
for column in object_columns_with_missing_values:
    mode_value = data[column].mode().iloc[0]  # Compute the mode for the column
    
    # Replace missing values with the mode
    data[column].fillna(mode_value, inplace=True)


# In[63]:


# Drop columns with more than 70% missing data
threshold = 0.7  # Set the threshold for missing data percentage
missing_data_percentage = data.isnull().mean()  # Compute the missing data percentage for each column
columns_to_drop = missing_data_percentage[missing_data_percentage > threshold].index  # Get columns with missing data above the threshold
data.drop(columns_to_drop, axis=1, inplace=True)  # Drop the columns with more than 70% missing data


# In[64]:


data.isnull().sum()


# In[65]:


# If we have to, we can drop the rest of the rows with missing values
data.dropna(inplace=True)


# # Data Cleaning and Preprocessing

# In[66]:


# Check for duplicates in the dataset
duplicates = data.duplicated()
print("Number of duplicates:", duplicates.sum())


# In[67]:


# Show the duplicate rows (none here)
duplicate_rows = data[duplicates]
print("Duplicate rows:")
print(duplicate_rows)


# In[68]:


# if there had been any duplicates, this is how to remove them
data.drop_duplicates(inplace=True)


# # Data Visualization:
# 

# In[69]:


plt.hist(data['SalePrice'], bins=50)
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.title('Histogram of Sale Price')
plt.show()


# In[70]:


sns.boxplot(x=data['OverallQual'],y=data['SalePrice'])
plt.xlabel('Overall Quality')
plt.ylabel('Sale Price')
plt.title('Box Plot of Sale Price by Overall Quality')
plt.show()


# In[71]:


# Scatter  plot
plt.scatter(data['GrLivArea'], data['SalePrice'])
plt.xlabel('Above Ground Living Area (sqft)')
plt.ylabel('Sale Price')
plt.title('Scatter Plot: Above Ground Living Area vs. Sale Price')
plt.show()


# # Univariate Analysis

# In[72]:


# Kernel Density Estimation (KDE) plot
sns.kdeplot(data['GrLivArea'], fill=True)
plt.xlabel('Above Ground Living Area (sqft)')
plt.ylabel('Density')
plt.title('KDE Plot: Above Ground Living Area')
plt.show()


# In[73]:


# Box Plot with outliers shown
plt.boxplot(data['GrLivArea'], showfliers=True)
plt.xlabel('Above Ground Living Area (sqft)')
plt.title('Box Plot with Outliers: Above Ground Living Area')
plt.show()


# # Multivariate Analysis

# In[74]:


# Multivariate Analysis
plt.bar(data['MSZoning'].value_counts().index, data['MSZoning'].value_counts().values)
plt.xlabel('MSZoning')
plt.ylabel('Count')
plt.title('Bar Chart of MSZoning')
plt.show()


# In[75]:


# Box Plot with different colors for each category
sns.boxplot(x='MSZoning', y='GrLivArea', data=data, hue='MSZoning')
plt.xlabel('MSZoning')
plt.ylabel('Above Ground Living Area (sqft)')
plt.title('Box Plot with Different Colors: MSZoning vs. Above Ground Living Area')
plt.show()


# In[76]:


# Swarm Plot
sns.swarmplot(x='MSZoning', y='GrLivArea', data=data)
plt.xlabel('MSZoning')
plt.ylabel('Above Ground Living Area (sqft)')
plt.title('Swarm Plot: MSZoning vs. Above Ground Living Area')
plt.show()


# In[77]:


sns.stripplot(x='MSZoning', y='GrLivArea', data=data)
plt.xlabel('MSZoning')
plt.ylabel('Above Ground Living Area (sqft)')
plt.title('Stripplot: MSZoning vs. Above Ground Living Area')
plt.show()


# In[78]:


# Violin plots
sns.violinplot(x='MSZoning', y='GrLivArea', data=data)
plt.xlabel('MSZoning')
plt.ylabel('Above Ground Living Area (sqft)')
plt.title('Violin Plot: MSZoning vs. Above Ground Living Area')
plt.show()


# In[79]:


# Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))  # Adjust the values to increase or decrease the figure size
plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Matrix')
plt.show()


# In[80]:


# Stacked Bar Chart
cross_tab = pd.crosstab(data['MSZoning'], data['SaleCondition'])
cross_tab.plot(kind='bar', stacked=True)
plt.xlabel('MSZoning')
plt.ylabel('Count')
plt.title('Stacked Bar Chart: MSZoning vs. SaleCondition')
plt.show()


# In[81]:


sns.scatterplot(x='LotArea', y='SalePrice', data=data)
plt.xlabel('Lot Area')
plt.ylabel('Sale Price')
plt.title('Scatter Plot of Lot Area vs. Sale Price')
plt.show()


# In[93]:


# Heatmap
plt.figure(figsize=(10, 8))  # Adjust the values to increase or decrease the figure size
sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
plt.title('Heatmap')
plt.show()


# # Feature Engineering

# In[84]:


# Feature Engineering
data['TotalBath'] = data['FullBath'] + data['HalfBath']
data['HasGarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)


# In[85]:


# Select categorical columns for encoding
categorical_columns = data.select_dtypes(include='object').columns

# Perform one-hot encoding
encoded_data = pd.get_dummies(data, columns=categorical_columns)


# # Statistical Analysis

# In[86]:


group1 = data[data['OverallQual'] >= 7]['SalePrice']
group2 = data[data['OverallQual'] < 7]['SalePrice']
t_statistic, p_value = ttest_ind(group1, group2)
print('T-Statistic:', t_statistic)
print('P-Value:', p_value)


# In[117]:


# Chi-Square Test
chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(data['MSZoning'], data['CentralAir']))
print('Chi-Square:', chi2)
print('P-Value:', p)


# # Outlier Detection:

# In[118]:


columns_to_check = ['LotArea', 'OverallQual', 'YearBuilt', 'GrLivArea']

plt.figure(figsize=(10, 8))  # Adjust the figure size if needed

for i, column in enumerate(columns_to_check):
    plt.subplot(2, 2, i+1)  # Create subplots for each column
    sns.boxplot(data[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)


# In[119]:


# Handling outliers by winsorizing
from scipy.stats import mstats
data['LotArea'] = mstats.winsorize(data['LotArea'], limits=[0.05, 0.05])


# In[120]:


sns.boxplot(data['LotArea'])
plt.xlabel('Lot Area')
plt.title('Box Plot of Lot Area')
plt.show()


# In[121]:


x = data.drop("SalePrice", axis=1)
y = data["SalePrice"]


# In[122]:


x = pd.get_dummies(x, column)


# In[123]:


x.head(4)


# In[124]:


y.head(3)


# In[125]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


# In[126]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[127]:


print('Shape for training data', x_train.shape, y_train.shape)
print("_________________________________")
print('Shape for testing data', x_test.shape, y_test.shape)


# In[130]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# # Random Forest Regressor Model .. acc 89%

# In[136]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=1000)


# In[137]:


model.fit(x_train,y_train)


# In[138]:


print("Accuracy --> ", model.score(x_test, y_test)*100)


# # Gradient Boosting Regressor Model .. acc 87%

# In[141]:


from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)


# In[142]:


model.fit(x_train,y_train)


# In[143]:


print("Accuracy --> ", model.score(x_test, y_test)*100)


# In[ ]:




