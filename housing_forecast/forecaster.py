import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



#hi i used lab 5 for reference for the preprocessing, feel free to add or change anything you want
#loading data for both years
df_2024=pd.read_csv('data/2024_data.csv')
df_2019=pd.read_csv('data/2019_data.csv')

print("rows:", len(df_2019))


feature_cols = ['Estimate!!MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS','Estimate!!MEDIAN GROSS RENT','Estimate!!HOUSING OCCUPANCY!!Total housing units!!Vacant housing units', "Estimate!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Bachelor's degree or higher"]

income_col = 'Estimate!!Households!!Median income (dollars)' #how much people can afford to pay
rent_col = 'Estimate!!GROSS RENT!!Occupied units paying rent!!Median (dollars)'  #local rental market/housing demand
vacant_col = 'Estimate!!HOUSING OCCUPANCY!!Total housing units!!Vacant housing units' #supply and availability
house_price_col = 'Estimate!!VALUE!!Owner-occupied units!!Median (dollars)'  
edu_col = "Estimate!!Total!!AGE BY EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Bachelor's degree or higher" #income and neighborhood desirability

df_2019_simple = df_2019[[income_col, rent_col, vacant_col, edu_col, house_price_col]]
df_2024_simple = df_2024[[income_col, rent_col, vacant_col, edu_col, house_price_col]] #remove any rows that have missing data

# #replacing missing values with mean 
# df_2024.fillna(df_2024.mean(numeric_only=True),inplace=True) 
# df_2019.fillna(df_2019.mean(numeric_only=True),inplace=True)

# #inputs and target separation
# inputs_2024, target_2024= df_2024.drop('target', axis=1),df_2024['target'] #target is not being found
# inputs_2019, target_2019= df_2019.drop('target', axis=1),df_2019['target']

inputs_2019 = df_2019_simple[[income_col, rent_col, vacant_col, edu_col]]
target_2019 = df_2019_simple[house_price_col]
inputs_2024 = df_2024_simple[[income_col, rent_col, vacant_col, edu_col]]
target_2024 = df_2024_simple[house_price_col]


#not splitting the data because we just have one row(just using dataset of 2019)
X_train = inputs_2019
y_train = target_2019 

#scaling features
scaler= StandardScaler()
scaler.fit(X_train)
inputs_2024_scaled= scaler.transform(inputs_2024)
inputs_2019_scaled= scaler.transform(inputs_2019)

# train on 2019
model = RandomForestRegressor(n_estimators=100,random_state=0)
model.fit(inputs_2019_scaled, target_2019)

#predicting and evaluating the results on 2024
y_pred2024 = model.predict(inputs_2024_scaled)
mse_2024 = mean_squared_error(target_2024, y_pred2024)
print("2019--2024 MSE:", mse_2024)
print("5 Predictions vs Actual:") # to check the five prediction
print("Pred:", y_pred2024[:5])
print("Real:", target_2024.values[:5])

