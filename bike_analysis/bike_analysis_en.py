"""
Bike Sharing Data Analysis and Regression Model Construction
"""
import os
os.environ['PYTHONUTF8'] = '1'

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# ================================================================
# 1. Load and Explore Data
# ================================================================
print("=" * 80)
print("1. Loading and Initial Data Exploration")
print("=" * 80)

df = pd.read_csv('bike-day.csv')
print(f"Dataset shape: {df.shape}")
print(f"Dataset size: {df.shape[0]} rows x {df.shape[1]} columns")

print("\nFirst 5 rows:")
print(df.head())

print("\nData info:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

# ================================================================
# 2. Data Cleaning
# ================================================================
print("\n" + "=" * 80)
print("2. Data Cleaning")
print("=" * 80)

print("\nMissing values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values")

print(f"\nDuplicate values: {df.duplicated().sum()}")

print("\nOutlier detection:")
numeric_columns = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    print(f"{col}: {outliers} outliers")

# Handle humidity outliers
print(f"\nDays with humidity=0: {df[df['hum'] == 0].shape[0]}")
df['hum'] = df['hum'].replace(0, np.nan)
df['hum'].fillna(df['hum'].median(skipna=True), inplace=True)
print("Fixed humidity=0 outliers with median")

# Ensure no NaN values remain
if df['hum'].isnull().any():
    df = df.dropna(subset=['hum'])
print(f"Remaining NaN in humidity: {df['hum'].isnull().sum()}")

# Handle windspeed outliers
print(f"\nMax windspeed: {df['windspeed'].max()}")
df['windspeed'] = df['windspeed'].clip(upper=df['windspeed'].quantile(0.99))
print(f"Max windspeed after clipping: {df['windspeed'].max()}")

# ================================================================
# 3. Exploratory Data Analysis
# ================================================================
print("\n" + "=" * 80)
print("3. Exploratory Data Analysis")
print("=" * 80)

# Create visualization charts
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Bike Sharing Data Exploration', fontsize=16, fontweight='bold')

# 3.1 Target variable distribution
axes[0, 0].hist(df['cnt'], bins=30, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Total Users Distribution')
axes[0, 0].set_xlabel('Number of Users')
axes[0, 0].set_ylabel('Frequency')

# 3.2 Season vs Users
season_names = ['Spring', 'Summer', 'Fall', 'Winter']
season_avg = df.groupby('season')['cnt'].mean()
axes[0, 1].bar(range(4), season_avg.values, color=['green', 'red', 'orange', 'blue'])
axes[0, 1].set_xticks(range(4))
axes[0, 1].set_xticklabels(season_names)
axes[0, 1].set_title('Average Users by Season')
axes[0, 1].set_ylabel('Average Users')

# 3.3 Month vs Users
monthly_avg = df.groupby('mnth')['cnt'].mean()
axes[0, 2].plot(monthly_avg.index, monthly_avg.values, marker='o', color='purple')
axes[0, 2].set_title('Average Users by Month')
axes[0, 2].set_xlabel('Month')
axes[0, 2].set_ylabel('Average Users')
axes[0, 2].grid(True, alpha=0.3)

# 3.4 Temperature vs Users
axes[1, 0].scatter(df['temp'], df['cnt'], alpha=0.5, color='orange')
axes[1, 0].set_title('Temperature vs Users')
axes[1, 0].set_xlabel('Temperature (normalized)')
axes[1, 0].set_ylabel('Users')

# 3.5 Humidity vs Users
axes[1, 1].scatter(df['hum'], df['cnt'], alpha=0.5, color='blue')
axes[1, 1].set_title('Humidity vs Users')
axes[1, 1].set_xlabel('Humidity (normalized)')
axes[1, 1].set_ylabel('Users')

# 3.6 Windspeed vs Users
axes[1, 2].scatter(df['windspeed'], df['cnt'], alpha=0.5, color='green')
axes[1, 2].set_title('Windspeed vs Users')
axes[1, 2].set_xlabel('Windspeed (normalized)')
axes[1, 2].set_ylabel('Users')

# 3.7 Workingday vs Weekend
workingday_avg = df.groupby('workingday')['cnt'].mean()
axes[2, 0].bar(['Weekend/Holiday', 'Workday'], workingday_avg.values, color=['coral', 'steelblue'])
axes[2, 0].set_title('Users: Workday vs Weekend')
axes[2, 0].set_ylabel('Average Users')

# 3.8 Weather Condition
weather_names = ['Sunny', 'Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow']
weather_avg = df.groupby('weathersit')['cnt'].mean()
axes[2, 1].bar(range(len(weather_avg)), weather_avg.values, color=['gold', 'lightblue', 'lightgray', 'darkgray'])
axes[2, 1].set_xticks(range(len(weather_avg)))
axes[2, 1].set_xticklabels([weather_names[int(x)-1] for x in weather_avg.index])
axes[2, 1].set_title('Users by Weather Condition')
axes[2, 1].set_ylabel('Average Users')

# 3.9 Year Comparison
yr_avg = df.groupby('yr')['cnt'].mean()
axes[2, 2].bar(['2011', '2012'], yr_avg.values, color=['lightgreen', 'lightcoral'])
axes[2, 2].set_title('Users by Year')
axes[2, 2].set_ylabel('Average Users')

plt.tight_layout()
plt.savefig('exploratory_analysis.png', dpi=300, bbox_inches='tight')
print("Exploratory analysis chart saved as 'exploratory_analysis.png'")

# 3.10 Correlation Analysis
plt.figure(figsize=(12, 10))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Correlation heatmap saved as 'correlation_heatmap.png'")

print("\nFeature correlation matrix:")
print(correlation_matrix)

# ================================================================
# 4. Feature Engineering
# ================================================================
print("\n" + "=" * 80)
print("4. Feature Engineering")
print("=" * 80)

feature_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                'weathersit', 'temp', 'hum', 'windspeed']

X = df[feature_cols].copy()
y = df['cnt'].copy()

print(f"\nNumber of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")

print("\nFinal features:", list(X.columns))

# ================================================================
# 5. Data Split
# ================================================================
print("\n" + "=" * 80)
print("5. Data Split")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['season']
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

# Standardize numeric features
numeric_features = ['temp', 'hum', 'windspeed']
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print("\nFeature standardization completed")
print("Standardization statistics (training set):")
print(X_train_scaled[numeric_features].describe())

# ================================================================
# 6. Model Building and Training
# ================================================================
print("\n" + "=" * 80)
print("6. Model Building and Training")
print("=" * 80)

# 6.1 Linear Regression
print("\n6.1 Linear Regression")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

y_train_pred_lr = lr_model.predict(X_train_scaled)
y_test_pred_lr = lr_model.predict(X_test_scaled)

print(f"Linear regression model trained")
print(f"Training R2: {r2_score(y_train, y_train_pred_lr):.4f}")
print(f"Test R2: {r2_score(y_test, y_test_pred_lr):.4f}")

print("\nLinear regression coefficients:")
coef_df = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Coefficient': lr_model.coef_
})
coef_df['Absolute Value'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('Absolute Value', ascending=False)
print(coef_df.to_string(index=False))

# 6.2 Random Forest Regression
print("\n6.2 Random Forest Regression")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

y_train_pred_rf = rf_model.predict(X_train_scaled)
y_test_pred_rf = rf_model.predict(X_test_scaled)

print(f"Random forest model trained")
print(f"Training R2: {r2_score(y_train, y_train_pred_rf):.4f}")
print(f"Test R2: {r2_score(y_test, y_test_pred_rf):.4f}")

print("\nRandom forest feature importance:")
feature_importance = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print(feature_importance.to_string(index=False))

# ================================================================
# 7. Model Evaluation
# ================================================================
print("\n" + "=" * 80)
print("7. Model Evaluation")
print("=" * 80)

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  R-squared (R2): {r2:.4f}")

    return {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

lr_metrics_train = evaluate_model(y_train, y_train_pred_lr, "Linear Regression (Training)")
lr_metrics_test = evaluate_model(y_test, y_test_pred_lr, "Linear Regression (Test)")
rf_metrics_train = evaluate_model(y_train, y_train_pred_rf, "Random Forest (Training)")
rf_metrics_test = evaluate_model(y_test, y_test_pred_rf, "Random Forest (Test)")

print("\n" + "=" * 80)
print("Model Performance Comparison")
print("=" * 80)

performance_comparison = pd.DataFrame([
    lr_metrics_test,
    rf_metrics_test
])
print(performance_comparison.to_string(index=False))

# ================================================================
# 8. Visualize Model Results
# ================================================================
print("\n" + "=" * 80)
print("8. Model Results Visualization")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')

# 8.1 Linear Regression: Predicted vs Actual (Training)
axes[0, 0].scatter(y_train, y_train_pred_lr, alpha=0.5, color='blue')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title('Linear Regression: Training Set')
axes[0, 0].grid(True, alpha=0.3)

# 8.2 Linear Regression: Predicted vs Actual (Test)
axes[0, 1].scatter(y_test, y_test_pred_lr, alpha=0.5, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual')
axes[0, 1].set_ylabel('Predicted')
axes[0, 1].set_title('Linear Regression: Test Set')
axes[0, 1].grid(True, alpha=0.3)

# 8.3 Linear Regression: Residual Analysis
residuals_lr = y_test - y_test_pred_lr
axes[0, 2].scatter(y_test_pred_lr, residuals_lr, alpha=0.5, color='purple')
axes[0, 2].axhline(y=0, color='r', linestyle='--')
axes[0, 2].set_xlabel('Predicted')
axes[0, 2].set_ylabel('Residuals')
axes[0, 2].set_title('Linear Regression: Residuals')
axes[0, 2].grid(True, alpha=0.3)

# 8.4 Random Forest: Predicted vs Actual (Training)
axes[1, 0].scatter(y_train, y_train_pred_rf, alpha=0.5, color='blue')
axes[1, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual')
axes[1, 0].set_ylabel('Predicted')
axes[1, 0].set_title('Random Forest: Training Set')
axes[1, 0].grid(True, alpha=0.3)

# 8.5 Random Forest: Predicted vs Actual (Test)
axes[1, 1].scatter(y_test, y_test_pred_rf, alpha=0.5, color='green')
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual')
axes[1, 1].set_ylabel('Predicted')
axes[1, 1].set_title('Random Forest: Test Set')
axes[1, 1].grid(True, alpha=0.3)

# 8.6 Random Forest: Residual Analysis
residuals_rf = y_test - y_test_pred_rf
axes[1, 2].scatter(y_test_pred_rf, residuals_rf, alpha=0.5, color='purple')
axes[1, 2].axhline(y=0, color='r', linestyle='--')
axes[1, 2].set_xlabel('Predicted')
axes[1, 2].set_ylabel('Residuals')
axes[1, 2].set_title('Random Forest: Residuals')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("Model performance visualization saved as 'model_performance.png'")

# Feature importance visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='viridis')
plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("Feature importance chart saved as 'feature_importance.png'")

# ================================================================
# 9. Generate Analysis Report
# ================================================================
print("\n" + "=" * 80)
print("9. Generate Analysis Report")
print("=" * 80)

report_content = f"""
# Bike Sharing Data Analysis and Regression Model Report

## Project Overview
This report provides a comprehensive analysis of bike sharing data, including data cleaning, exploratory analysis, feature engineering, model construction, and evaluation.

## 1. Dataset Information
- Dataset size: {df.shape[0]} rows × {df.shape[1]} columns
- Time range: {df['dteday'].min()} to {df['dteday'].max()}
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
- Mean: {df['cnt'].mean():.2f}
- Median: {df['cnt'].median():.2f}
- Standard deviation: {df['cnt'].std():.2f}
- Min: {df['cnt'].min()}, Max: {df['cnt'].max()}

### 3.2 Key Findings
1. Season effect: Fall and Summer have the most users
2. Temperature relationship: Positive correlation with users
3. Workday effect: Slightly more users on workdays
4. Weather impact: Most users on sunny days, least on bad weather
5. Year growth: Significant increase from 2011 to 2012

### 3.3 Feature Correlation
- temp vs cnt: {correlation_matrix.loc['temp', 'cnt']:.3f}
- hum vs cnt: {correlation_matrix.loc['hum', 'cnt']:.3f}
- windspeed vs cnt: {correlation_matrix.loc['windspeed', 'cnt']:.3f}

## 4. Model Construction

### 4.1 Data Split
- Training set: {X_train.shape[0]} samples (80%)
- Test set: {X_test.shape[0]} samples (20%)
- Stratified by season

### 4.2 Linear Regression
Parameters:
- Algorithm: Ordinary Least Squares
- Features: {len(X_train_scaled.columns)}
- Samples: {X_train_scaled.shape[0]}

Top Coefficients:
{coef_df.head(5).to_string(index=False)}

### 4.3 Random Forest Regression
Parameters:
- n_estimators: 100
- max_depth: 10
- min_samples_split: 5
- min_samples_leaf: 2
- random_state: 42

Top 5 Feature Importance:
{feature_importance.head(5).to_string(index=False)}

## 5. Model Evaluation

### 5.1 Metrics
- MSE: Mean Squared Error (lower is better)
- RMSE: Root Mean Squared Error (lower is better)
- MAE: Mean Absolute Error (lower is better)
- R2: R-squared (closer to 1 is better)

### 5.2 Performance Comparison

{performance_comparison.to_string(index=False)}

### 5.3 Model Analysis

**Linear Regression:**
- Training R2: {r2_score(y_train, y_train_pred_lr):.4f}
- Test R2: {r2_score(y_test, y_test_pred_lr):.4f}
- Pros: Simple, highly interpretable
- Cons: Limited prediction accuracy, may miss nonlinear relationships

**Random Forest:**
- Training R2: {r2_score(y_train, y_train_pred_rf):.4f}
- Test R2: {r2_score(y_test, y_test_pred_rf):.4f}
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

Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Tools: Python (pandas, scikit-learn, matplotlib, seaborn)
"""

with open('bike_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("Detailed analysis report saved as 'bike_analysis_report.md'")

performance_comparison.to_csv('model_performance_comparison.csv', index=False, encoding='utf-8')
print("Model performance comparison saved as 'model_performance_comparison.csv'")

feature_importance.to_csv('feature_importance.csv', index=False, encoding='utf-8')
print("Feature importance saved as 'feature_importance.csv'")

print("\n" + "=" * 80)
print("Analysis completed!")
print("=" * 80)
print(f"Generated files:")
print(f"  1. exploratory_analysis.png - Exploratory analysis charts")
print(f"  2. correlation_heatmap.png - Feature correlation heatmap")
print(f"  3. model_performance.png - Model performance visualization")
print(f"  4. feature_importance.png - Feature importance ranking")
print(f"  5. bike_analysis_report.md - Detailed analysis report")
print(f"  6. model_performance_comparison.csv - Performance comparison")
print(f"  7. feature_importance.csv - Feature importance data")
print("=" * 80)