"""
简化版本的共享单车分析脚本，用于调试
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("开始加载数据...")
df = pd.read_csv('bike-day.csv')
print(f"数据加载完成: {df.shape}")

# 数据预处理
df['hum'] = df['hum'].replace(0, np.nan)
df['hum'].fillna(df['hum'].median(), inplace=True)

# 特征选择
feature_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                'weathersit', 'temp', 'hum', 'windspeed']
X = df[feature_cols].copy()
y = df['cnt'].copy()

print(f"特征变量: {list(X.columns)}")

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['season']
)

print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

# 特征标准化
numeric_features = ['temp', 'hum', 'windspeed']
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print("特征标准化完成")

# 线性回归模型
print("\n训练线性回归模型...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
print("线性回归模型训练完成")

y_train_pred_lr = lr_model.predict(X_train_scaled)
y_test_pred_lr = lr_model.predict(X_test_scaled)

print(f"线性回归 - 训练集 R2: {r2_score(y_train, y_train_pred_lr):.4f}")
print(f"线性回归 - 测试集 R2: {r2_score(y_test, y_test_pred_lr):.4f}")

# 随机森林回归模型
print("\n训练随机森林回归模型...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
print("随机森林回归模型训练完成")

y_train_pred_rf = rf_model.predict(X_train_scaled)
y_test_pred_rf = rf_model.predict(X_test_scaled)

print(f"随机森林 - 训练集 R2: {r2_score(y_train, y_train_pred_rf):.4f}")
print(f"随机森林 - 测试集 R2: {r2_score(y_test, y_test_pred_rf):.4f}")

# 模型评估
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        '模型': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

print("\n模型性能对比:")
lr_metrics = evaluate_model(y_test, y_test_pred_lr, "线性回归")
rf_metrics = evaluate_model(y_test, y_test_pred_rf, "随机森林")

metrics_df = pd.DataFrame([lr_metrics, rf_metrics])
print(metrics_df.to_string(index=False))

# 保存结果
metrics_df.to_csv('model_performance_comparison.csv', index=False, encoding='utf-8')
print("\n模型性能对比已保存为 'model_performance_comparison.csv'")

print("\n所有任务完成！")