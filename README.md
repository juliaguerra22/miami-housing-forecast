# Miami-housing-forecast

This project focuses on predicting housing prices in Miami-Dade County over the next five years, specifically by housing size and type. We initially planned to predict gentrification risk at the neighborhood level, but due to the lack of consistent neighborhood-level data for Miami-Dade and time constraints, we shifted focus to county-level housing price forecasting.

## Approach 
We will build a **Random Forest regression model**, chosen because prior research (Yoo, 2023) shows that Random Forests perform especially well in gentrification prediction tasks, which was our previous idea, but we expect it to perform just as well for this model. This model will make use of RF in order to find **nonlinear relationships** between **housing prices and factors like income, rent, vacancy rates, and education levels**.

## Dataset

**-** The dataset is created by combining multiple public sources from the **American Community Survey (ACS)** for Miami-Dade County from **2019** and **2024**.
**- This data includes:** Median rents, home values, income levels, vacancy rates, poverty rates, and education attainment all at the county level
**-** After preprocessing and cleaning, the dataset includes:
  **-** Predictive variables (features)

## Features
The key features used to predict housing prices include:
- Median rent  
- Median household income  
- Home values  
- Vacancy rates  
- Poverty rate  
- Levels of educational attainment  

## Model Evaluation
We will evaluate model performance primarily using regression metrics such as:
- Mean Squared Error **(MSE)**
- Mean Absolute Error **(MAE)**




