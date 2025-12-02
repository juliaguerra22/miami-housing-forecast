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

print("2024 columns:", df_2024.columns)
print("2019 columns:", df_2019.columns)

#replacing missing values with mean 
df_2024.fillna(df_2024.mean(numeric_only=True),inplace=True) 
df_2019.fillna(df_2019.mean(numeric_only=True),inplace=True)

#inputs and target separation
inputs_2024, target_2024= df_2024.drop('target', axis=1),df_2024['target']
inputs_2019, target_2019= df_2019.drop('target', axis=1),df_2019['target']

#scaling features
scaler= StandardScaler()
inputs_2024_scaled= scaler.fit_transform(inputs_2024)
inputs_2019_scaled= scaler.fit_transform(inputs_2019)
scaled_2024= pd.DataFrame(inputs_2024_scaled, columns=inputs_2024.columns)
scaled_2019= pd.DataFrame(inputs_2019_scaled, columns=inputs_2019.columns)

#data split 
X_train_2024, X_test_2024, y_train_2024, y_test_2024= train_test_split(scaled_2024, target_2024, test_size=0.2, random_state=42)
X_train_2019, X_test_2019, y_train_2019, y_test_2019= train_test_split(scaled_2019, target_2019, test_size=0.2, random_state=42)

#traning 
model_2024= RandomForestRegressor(n_estimators=100, random_state=42)
model_2024.fit(X_train_2024, y_train_2024)
model_2019= RandomForestRegressor(n_estimators=100, random_state=42)
model_2019.fit(X_train_2019, y_train_2019)

#preds
preds_2024=model_2024.predict(X_test_2024)
preds_2019=model_2019.predict(X_test_2019)

#evaluation
mse_2024= mean_squared_error(y_test_2024, preds_2024)
mse_2019= mean_squared_error(y_test_2019, preds_2019)

print(f"Mean Squared Error for 2024 data: {mse_2024}")
print(f"Mean Squared Error for 2019 data: {mse_2019}")