
# Bike Sharing Data Analysis and Regression Model Report

## Project Overview
This report provides a comprehensive analysis of bike sharing data, including data cleaning, exploratory analysis, feature engineering, model construction, and evaluation.

## 1. Dataset Information
- Dataset size: 730 rows × 16 columns
- Time range: 2011/1/1 to 2012/9/9
- Target variable: cnt (total users)

### Feature Description:
- season: Season (1:Spring, 2:Summer, 3:Fall, 4:Winter)
- yr: Year (0:2011, 1:2012)
- mnth: Month (1-12)
- holiday: Holiday flag (0:No, 1:Yes)
- weekday: Day of week (0-6)
- workingday: Working day flag (0:No, 1:Yes)
- weathersit: Weather condition (1:Sunny, 2:Cloudy, 3:Light Rain/Snow, 4:Heavy Rain/Snow)
- temp: Temperature (normalized)
- hum: Humidity (normalized)
- windspeed: Windspeed (normalized)

## 2. Data Preprocessing

### 2.1 Missing Values
- No missing values in original data
- Fixed humidity=0 outliers with median

### 2.2 Outlier Handling
- Detected outliers using IQR method
- Clipped windspeed at 99th percentile

### 2.3 Feature Standardization
- Standardized numeric features
- Features: temp, hum, windspeed

## 3. Exploratory Data Analysis

### 3.1 Target Variable Distribution
- Mean: 4509.67
- Median: 4548.50
- Standard deviation: 1933.19
- Min: 22, Max: 8714

### 3.2 Key Findings
1. Season effect: Fall and Summer have the most users
2. Temperature relationship: Positive correlation with users
3. Workday effect: Slightly more users on workdays
4. Weather impact: Most users on sunny days, least on bad weather
5. Year growth: Significant increase from 2011 to 2012

### 3.3 Feature Correlation
- temp vs cnt: 0.628
- hum vs cnt: -0.115
- windspeed vs cnt: -0.232

## 4. Model Construction

### 4.1 Data Split
- Training set: 584 samples (80%)
- Test set: 146 samples (20%)
- Stratified by season

### 4.2 Linear Regression
Parameters:
- Algorithm: Ordinary Least Squares
- Features: 10
- Samples: 584

Top Coefficients:
   Feature  Coefficient  Absolute Value
        yr  2021.021024     2021.021024
      temp   944.216292      944.216292
   holiday  -574.332876      574.332876
weathersit  -538.859532      538.859532
    season   484.884145      484.884145

### 4.3 Random Forest Regression
Parameters:
- n_estimators: 100
- max_depth: 10
- min_samples_split: 5
- min_samples_leaf: 2
- random_state: 42

Top 5 Feature Importance:
Feature  Importance
   temp    0.485067
     yr    0.307001
 season    0.057774
    hum    0.054940
   mnth    0.031677

## 5. Model Evaluation

### 5.1 Metrics
- MSE: Mean Squared Error (lower is better)
- RMSE: Root Mean Squared Error (lower is better)
- MAE: Mean Absolute Error (lower is better)
- R2: R-squared (closer to 1 is better)

### 5.2 Performance Comparison

                   Model           MSE       RMSE        MAE       R2
Linear Regression (Test) 787259.919739 887.276687 661.107166 0.784976
    Random Forest (Test) 428413.242199 654.532843 451.387638 0.882988

### 5.3 Model Analysis

**Linear Regression:**
- Training R2: 0.8008
- Test R2: 0.7850
- Pros: Simple, highly interpretable
- Cons: Limited prediction accuracy, may miss nonlinear relationships

**Random Forest:**
- Training R2: 0.9616
- Test R2: 0.8830
- Pros: High prediction accuracy, captures nonlinear relationships
- Cons: Higher complexity, less interpretable

## 6. Conclusion and Recommendations

### 6.1 Key Conclusions
1. Random Forest outperforms Linear Regression significantly
2. Temperature and year are the most important factors
3. Weather and season also significantly impact user count

### 6.2 Model Selection
- For prediction accuracy: Random Forest is recommended
- For interpretability: Linear Regression is suitable

### 6.3 Improvement Directions
1. Try other algorithms (XGBoost, LightGBM)
2. Deepen feature engineering
3. Consider time series features
4. Hyperparameter tuning

## 7. Visualization Charts
- exploratory_analysis.png - Exploratory data analysis
- correlation_heatmap.png - Feature correlation heatmap
- model_performance.png - Model performance visualization
- feature_importance.png - Feature importance ranking

---

Report generated: 2026-06-30 10:52:35
Tools: Python (pandas, scikit-learn, matplotlib, seaborn)
