#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np # for array operations
import pandas as pd # for working with DataFrames
import seaborn as sns #for the plot
import matplotlib.pyplot as plt # for data visualization

from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.linear_model import LinearRegression #for linear regression 
from sklearn.ensemble import RandomForestRegressor # for building the RFR model


# In[44]:


df = pd.read_json('C:/Users/bianc/OneDrive/Dokumenter/Exchange/ADA/Project/data_all_plus_good.json') #read the file (json)
df.columns  # print collums of the dataframe


# In[45]:


# Data exploring

df.head() # Print first 5 points


# In[46]:


df.describe() # print describtive statistics for each feature


# In[47]:


# Creating plots for exploring the data

# Select the column for "no grad 1000"
no_grad_values = df['no grad pr 1000']
country = df['country']
ai_enter_values_column = df['ai_enter']

# Create a histogram from bar plot
plt.figure(figsize=(14, 6))
sns.barplot(x=country, y=no_grad_values, data=df, palette='viridis')

# Add labels and title
plt.xlabel('European countries')
plt.ylabel('Number of women STEM graduates (per 1000 people)')
plt.title('Women STEM graduates across European countries (2021)')

# Rotating x-axis labels for better readability
plt.xticks(rotation=90)

# Showing the plot of women in stem grad
plt.show()

# Create a bar plot for AI enterprise
plt.figure(figsize=(14, 6))
sns.barplot(x=country, y=ai_enter_values_column, data=df, palette='viridis')

# Add labels and title
plt.xlabel('European countries')
plt.ylabel('Level of AI Use in Enterprises')
plt.title('Level of AI Use in Enterprises across European countries')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Show the plot
plt.show()


# In[48]:


# Data preprocessing 

# Selecting feature variables
features = df[['ai_enter','internet use','internet access','E-Government','good digital skills','NOP never used internet']]

# Picking target variable
target = df['no grad pr 1000']

# Train-test - splitting dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# In[49]:


# Creating linear regression model
linear_reg_model = LinearRegression()

# Training the model
linear_reg_model.fit(X_train, y_train)

# Making predictions on the test set
y_linear_pred = linear_reg_model.predict(X_test)

# Printing the RMSE and MEA 
rmse = np.sqrt(mean_squared_error(y_linear_pred, y_test))
MAE = mean_absolute_error(y_test, y_linear_pred)

print("Root Mean Squared Error: {}".format(rmse))
print('Mean Absolute Error : ', MAE) 

# Plot the scatter plot with the linear regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_linear_pred)

# Adding labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Actual vs Predicted Values with Linear Regression Line')

# Add a linear regression line
sns.regplot(x=y_test, y=y_linear_pred, scatter=False, color='red', line_kws={'linewidth': 2})

# Show plot
plt.show()


# In[50]:


# Extra: 
# Creating Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Training the model
rf_model.fit(X_train, y_train)

# Making the predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Feature Importance to show importence of each feature for the model
feature_importance = pd.Series(rf_model.feature_importances_, index=features.columns)
feature_importance.plot(kind='barh')
plt.title('Feature Importance graph')

#Show plot
plt.show()


# In[ ]:




