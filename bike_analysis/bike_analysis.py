"""
共享单车数据分析与回归模型构建
完成：数据清洗、探索性分析、特征工程、模型构建、性能评估及报告生成
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

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

# 设置中文字体和显示风格
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# ==============================================================================
# 1. 数据加载与初步探索
# ==============================================================================
print("=" * 80)
print("1. 数据加载与初步探索")
print("=" * 80)

# 加载数据
df = pd.read_csv('bike-day.csv')
print(f"数据集形状: {df.shape}")
print(f"数据集大小: {df.shape[0]} 行 × {df.shape[1]} 列")

# 显示前几行数据
print("\n前5行数据:")
print(df.head())

# 数据基本信息
print("\n数据基本信息:")
print(df.info())

# 统计描述
print("\n统计描述:")
print(df.describe())

# ==============================================================================
# 2. 数据清洗
# ==============================================================================
print("\n" + "=" * 80)
print("2. 数据清洗")
print("=" * 80)

# 检查缺失值
print("\n缺失值统计:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if any(missing_values > 0) else "无缺失值")

# 检查重复值
print(f"\n重复值数量: {df.duplicated().sum()}")

# 检查异常值（使用IQR方法）
print("\n异常值检测:")
numeric_columns = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    print(f"{col}: {outliers} 个异常值")

# 特殊处理：湿度值为0的异常值
print(f"\n湿度为0的天数: {df[df['hum'] == 0].shape[0]}")
df['hum'] = df['hum'].replace(0, np.nan)
df['hum'].fillna(df['hum'].median(), inplace=True)
print("已处理湿度为0的异常值，用中位数填充")

# 处理风速异常值
print(f"\n风速最大值: {df['windspeed'].max()}")
df['windspeed'] = df['windspeed'].clip(upper=df['windspeed'].quantile(0.99))
print(f"处理后风速最大值: {df['windspeed'].max()}")

# ==============================================================================
# 3. 探索性数据分析
# ==============================================================================
print("\n" + "=" * 80)
print("3. 探索性数据分析")
print("=" * 80)

# 创建可视化图表
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('共享单车数据探索性分析', fontsize=16, fontweight='bold')

# 3.1 目标变量分布
axes[0, 0].hist(df['cnt'], bins=30, color='skyblue', edgecolor='black')
axes[0, 0].set_title('总用户数量分布')
axes[0, 0].set_xlabel('用户数量')
axes[0, 0].set_ylabel('频次')

# 3.2 季节与用户数量
season_names = ['春季', '夏季', '秋季', '冬季']
season_avg = df.groupby('season')['cnt'].mean()
axes[0, 1].bar(range(4), season_avg.values, color=['green', 'red', 'orange', 'blue'])
axes[0, 1].set_xticks(range(4))
axes[0, 1].set_xticklabels(season_names)
axes[0, 1].set_title('各季节平均用户数量')
axes[0, 1].set_ylabel('平均用户数量')

# 3.3 月份与用户数量
monthly_avg = df.groupby('mnth')['cnt'].mean()
axes[0, 2].plot(monthly_avg.index, monthly_avg.values, marker='o', color='purple')
axes[0, 2].set_title('月份与平均用户数量')
axes[0, 2].set_xlabel('月份')
axes[0, 2].set_ylabel('平均用户数量')
axes[0, 2].grid(True, alpha=0.3)

# 3.4 温度与用户数量
axes[1, 0].scatter(df['temp'], df['cnt'], alpha=0.5, color='orange')
axes[1, 0].set_title('温度与用户数量')
axes[1, 0].set_xlabel('温度（归一化）')
axes[1, 0].set_ylabel('用户数量')

# 3.5 湿度与用户数量
axes[1, 1].scatter(df['hum'], df['cnt'], alpha=0.5, color='blue')
axes[1, 1].set_title('湿度与用户数量')
axes[1, 1].set_xlabel('湿度（归一化）')
axes[1, 1].set_ylabel('用户数量')

# 3.6 风速与用户数量
axes[1, 2].scatter(df['windspeed'], df['cnt'], alpha=0.5, color='green')
axes[1, 2].set_title('风速与用户数量')
axes[1, 2].set_xlabel('风速（归一化）')
axes[1, 2].set_ylabel('用户数量')

# 3.7 工作日与周末对比
workingday_avg = df.groupby('workingday')['cnt'].mean()
axes[2, 0].bar(['周末/节假日', '工作日'], workingday_avg.values, color=['coral', 'steelblue'])
axes[2, 0].set_title('工作日与周末用户数量对比')
axes[2, 0].set_ylabel('平均用户数量')

# 3.8 天气状况影响
weather_names = ['晴天', '多云', '小雨/雪', '恶劣天气']
weather_avg = df.groupby('weathersit')['cnt'].mean()
axes[2, 1].bar(range(len(weather_avg)), weather_avg.values, color=['gold', 'lightblue', 'lightgray', 'darkgray'])
axes[2, 1].set_xticks(range(len(weather_avg)))
axes[2, 1].set_xticklabels([weather_names[int(x)-1] for x in weather_avg.index])
axes[2, 1].set_title('天气状况与用户数量')
axes[2, 1].set_ylabel('平均用户数量')

# 3.9 年份对比
yr_avg = df.groupby('yr')['cnt'].mean()
axes[2, 2].bar(['2011年', '2012年'], yr_avg.values, color=['lightgreen', 'lightcoral'])
axes[2, 2].set_title('年份数据对比')
axes[2, 2].set_ylabel('平均用户数量')

plt.tight_layout()
plt.savefig('exploratory_analysis.png', dpi=300, bbox_inches='tight')
print("探索性分析图表已保存为 'exploratory_analysis.png'")

# 3.10 特征相关性分析
plt.figure(figsize=(12, 10))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": 0.8})
plt.title('特征相关性热力图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("相关性热力图已保存为 'correlation_heatmap.png'")

# 显示相关性矩阵
print("\n特征相关系数矩阵:")
print(correlation_matrix)

# ==============================================================================
# 4. 特征工程
# ==============================================================================
print("\n" + "=" * 80)
print("4. 特征工程")
print("=" * 80)

# 选择特征变量
# 移除instant（索引）、dteday（日期）、cnt（目标变量）
# 也移除casual和registered，因为它们是cnt的组成部分
feature_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

# 创建特征矩阵和目标变量
X = df[feature_cols].copy()
y = df['cnt'].copy()

print(f"\n特征变量数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")

# 特征选择：移除高度相关的特征
# temp和atemp相关性很高，只保留temp
X = X.drop(['atemp'], axis=1)
print("移除高度相关的特征: atemp")

print("\n最终特征变量:", list(X.columns))

# ==============================================================================
# 5. 数据集划分
# ==============================================================================
print("\n" + "=" * 80)
print("5. 数据集划分")
print("=" * 80)

# 按8:2比例划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['season']
)

print(f"训练集大小: {X_train.shape[0]} 样本 ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
print(f"测试集大小: {X_test.shape[0]} 样本 ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

# 标准化数值特征
numeric_features = ['temp', 'hum', 'windspeed']
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print("\n特征标准化完成")
print("标准化统计信息（训练集）:")
print(X_train_scaled[numeric_features].describe())

# ==============================================================================
# 6. 模型构建与训练
# ==============================================================================
print("\n" + "=" * 80)
print("6. 模型构建与训练")
print("=" * 80)

# 6.1 线性回归模型
print("\n6.1 线性回归模型")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# 线性回归模型评估
y_train_pred_lr = lr_model.predict(X_train_scaled)
y_test_pred_lr = lr_model.predict(X_test_scaled)

print(f"线性回归模型训练完成")
print(f"训练集 R2: {r2_score(y_train, y_train_pred_lr):.4f}")
print(f"测试集 R2: {r2_score(y_test, y_test_pred_lr):.4f}")

# 显示线性回归系数
print("\n线性回归模型系数:")
coef_df = pd.DataFrame({
    '特征': X_train_scaled.columns,
    '系数': lr_model.coef_
})
coef_df['绝对值'] = coef_df['系数'].abs()
coef_df = coef_df.sort_values('绝对值', ascending=False)
print(coef_df.to_string(index=False))

# 6.2 随机森林回归模型
print("\n6.2 随机森林回归模型")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

# 随机森林模型评估
y_train_pred_rf = rf_model.predict(X_train_scaled)
y_test_pred_rf = rf_model.predict(X_test_scaled)

print(f"随机森林回归模型训练完成")
print(f"训练集 R2: {r2_score(y_train, y_train_pred_rf):.4f}")
print(f"测试集 R2: {r2_score(y_test, y_test_pred_rf):.4f}")

# 显示特征重要性
print("\n随机森林特征重要性:")
feature_importance = pd.DataFrame({
    '特征': X_train_scaled.columns,
    '重要性': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('重要性', ascending=False)
print(feature_importance.to_string(index=False))

# ==============================================================================
# 7. 模型性能评估
# ==============================================================================
print("\n" + "=" * 80)
print("7. 模型性能评估")
print("=" * 80)

def evaluate_model(y_true, y_pred, model_name):
    """评估模型性能"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} 性能指标:")
    print(f"  均方误差 (MSE): {mse:.2f}")
    print(f"  均方根误差 (RMSE): {rmse:.2f}")
    print(f"  平均绝对误差 (MAE): {mae:.2f}")
    print(f"  决定系数 (R2): {r2:.4f}")

    return {
        '模型': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# 评估线性回归模型
lr_metrics_train = evaluate_model(y_train, y_train_pred_lr, "线性回归（训练集）")
lr_metrics_test = evaluate_model(y_test, y_test_pred_lr, "线性回归（测试集）")

# 评估随机森林模型
rf_metrics_train = evaluate_model(y_train, y_train_pred_rf, "随机森林（训练集）")
rf_metrics_test = evaluate_model(y_test, y_test_pred_rf, "随机森林（测试集）")

# 创建性能对比表格
print("\n" + "=" * 80)
print("模型性能对比")
print("=" * 80)

performance_comparison = pd.DataFrame([
    lr_metrics_test,
    rf_metrics_test
])
print(performance_comparison.to_string(index=False))

# ==============================================================================
# 8. 可视化模型结果
# ==============================================================================
print("\n" + "=" * 80)
print("8. 模型结果可视化")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('模型性能可视化分析', fontsize=16, fontweight='bold')

# 8.1 线性回归：预测值 vs 实际值（训练集）
axes[0, 0].scatter(y_train, y_train_pred_lr, alpha=0.5, color='blue')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('实际值')
axes[0, 0].set_ylabel('预测值')
axes[0, 0].set_title('线性回归：训练集预测 vs 实际')
axes[0, 0].grid(True, alpha=0.3)

# 8.2 线性回归：预测值 vs 实际值（测试集）
axes[0, 1].scatter(y_test, y_test_pred_lr, alpha=0.5, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('实际值')
axes[0, 1].set_ylabel('预测值')
axes[0, 1].set_title('线性回归：测试集预测 vs 实际')
axes[0, 1].grid(True, alpha=0.3)

# 8.3 线性回归：残差分析
residuals_lr = y_test - y_test_pred_lr
axes[0, 2].scatter(y_test_pred_lr, residuals_lr, alpha=0.5, color='purple')
axes[0, 2].axhline(y=0, color='r', linestyle='--')
axes[0, 2].set_xlabel('预测值')
axes[0, 2].set_ylabel('残差')
axes[0, 2].set_title('线性回归：残差分析')
axes[0, 2].grid(True, alpha=0.3)

# 8.4 随机森林：预测值 vs 实际值（训练集）
axes[1, 0].scatter(y_train, y_train_pred_rf, alpha=0.5, color='blue')
axes[1, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('实际值')
axes[1, 0].set_ylabel('预测值')
axes[1, 0].set_title('随机森林：训练集预测 vs 实际')
axes[1, 0].grid(True, alpha=0.3)

# 8.5 随机森林：预测值 vs 实际值（测试集）
axes[1, 1].scatter(y_test, y_test_pred_rf, alpha=0.5, color='green')
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('实际值')
axes[1, 1].set_ylabel('预测值')
axes[1, 1].set_title('随机森林：测试集预测 vs 实际')
axes[1, 1].grid(True, alpha=0.3)

# 8.6 随机森林：残差分析
residuals_rf = y_test - y_test_pred_rf
axes[1, 2].scatter(y_test_pred_rf, residuals_rf, alpha=0.5, color='purple')
axes[1, 2].axhline(y=0, color='r', linestyle='--')
axes[1, 2].set_xlabel('预测值')
axes[1, 2].set_ylabel('残差')
axes[1, 2].set_title('随机森林：残差分析')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("模型性能可视化图表已保存为 'model_performance.png'")

# 特征重要性可视化
plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=feature_importance.head(10), palette='viridis')
plt.title('随机森林模型 - 特征重要性排序', fontsize=14, fontweight='bold')
plt.xlabel('重要性分数')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("特征重要性图表已保存为 'feature_importance.png'")

# ==============================================================================
# 9. 生成分析报告
# ==============================================================================
print("\n" + "=" * 80)
print("9. 生成分析报告")
print("=" * 80)

report_content = f"""
# 共享单车数据分析与回归模型构建报告

## 项目概述
本报告对共享单车数据进行了全面的分析，包括数据清洗、探索性分析、特征工程、模型构建与评估。

## 1. 数据集信息
- 数据集大小: {df.shape[0]} 行 × {df.shape[1]} 列
- 时间范围: {df['dteday'].min()} 至 {df['dteday'].max()}
- 目标变量: cnt (总用户数量)

### 特征变量说明:
- season: 季节 (1:春季, 2:夏季, 3:秋季, 4:冬季)
- yr: 年份 (0:2011, 1:2012)
- mnth: 月份 (1-12)
- holiday: 是否节假日 (0:否, 1:是)
- weekday: 星期几 (0-6)
- workingday: 是否工作日 (0:否, 1:是)
- weathersit: 天气状况 (1:晴天, 2:多云, 3:小雨/雪, 4:恶劣天气)
- temp: 温度 (归一化值)
- atemp: 体感温度 (归一化值) - 已移除（与temp高度相关）
- hum: 湿度 (归一化值)
- windspeed: 风速 (归一化值)

## 2. 数据预处理

### 2.1 缺失值处理
- 原始数据无缺失值
- 处理了湿度为0的异常值，用中位数填充

### 2.2 异常值处理
- 使用IQR方法检测异常值
- 对风速进行99分位数截断处理

### 2.3 特征标准化
- 对数值特征进行标准化处理
- 标准化特征: temp, hum, windspeed

## 3. 探索性数据分析

### 3.1 目标变量分布
- 总用户数量均值为 {df['cnt'].mean():.2f}
- 中位数为 {df['cnt'].median():.2f}
- 标准差为 {df['cnt'].std():.2f}
- 最小值为 {df['cnt'].min()}，最大值为 {df['cnt'].max()}

### 3.2 关键发现
1. 季节影响: 秋季和夏季用户数量最多
2. 温度关系: 温度与用户数量呈正相关
3. 工作日效应: 工作日用户数量略高于周末
4. 天气影响: 晴天用户数量最多，恶劣天气最少
5. 年份增长: 2012年比2011年用户数量显著增加

### 3.3 特征相关性
- temp与cnt相关性: {correlation_matrix.loc['temp', 'cnt']:.3f}
- hum与cnt相关性: {correlation_matrix.loc['hum', 'cnt']:.3f}
- windspeed与cnt相关性: {correlation_matrix.loc['windspeed', 'cnt']:.3f}

## 4. 模型构建

### 4.1 数据集划分
- 训练集: {X_train.shape[0]} 样本 (80%)
- 测试集: {X_test.shape[0]} 样本 (20%)
- 按季节分层抽样，保持季节分布一致性

### 4.2 线性回归模型
模型参数:
- 算法: 普通最小二乘法
- 特征数量: {len(X_train_scaled.columns)}
- 样本数量: {X_train_scaled.shape[0]}

主要系数:
{coef_df.head(5).to_string(index=False)}

### 4.3 随机森林回归模型
模型参数:
- n_estimators: 100
- max_depth: 10
- min_samples_split: 5
- min_samples_leaf: 2
- random_state: 42

特征重要性 Top 5:
{feature_importance.head(5).to_string(index=False)}

## 5. 模型性能评估

### 5.1 评估指标
- MSE (均方误差): 越小越好
- RMSE (均方根误差): 越小越好
- MAE (平均绝对误差): 越小越好
- R2 (决定系数): 越接近1越好

### 5.2 性能对比

{performance_comparison.to_string(index=False)}

### 5.3 模型分析

**线性回归模型:**
- 训练集 R2: {r2_score(y_train, y_train_pred_lr):.4f}
- 测试集 R2: {r2_score(y_test, y_test_pred_lr):.4f}
- 优点: 模型简单，可解释性强
- 缺点: 预测精度有限，可能存在非线性关系未捕捉

**随机森林回归模型:**
- 训练集 R2: {r2_score(y_train, y_train_pred_rf):.4f}
- 测试集 R2: {r2_score(y_test, y_test_pred_rf):.4f}
- 优点: 预测精度高，能捕捉非线性关系
- 缺点: 模型复杂度较高，可解释性相对较弱

## 6. 结论与建议

### 6.1 主要结论
1. 随机森林模型的预测性能明显优于线性回归模型
2. 温度和年份是影响共享单车使用量的最重要因素
3. 天气状况和季节也对用户数量有显著影响

### 6.2 模型选择建议
- 如果追求预测精度，推荐使用随机森林模型
- 如果需要模型可解释性，可以考虑线性回归模型

### 6.3 改进方向
1. 尝试其他回归算法（如XGBoost、LightGBM）
2. 进行更深入的特征工程
3. 考虑时间序列特征
4. 进行超参数调优

## 7. 可视化图表
本分析生成了以下可视化图表:
1. exploratory_analysis.png - 探索性数据分析图表
2. correlation_heatmap.png - 特征相关性热力图
3. model_performance.png - 模型性能可视化
4. feature_importance.png - 特征重要性排序

---

报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
分析工具: Python (pandas, scikit-learn, matplotlib, seaborn)
"""

# 保存报告
with open('bike_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("详细分析报告已保存为 'bike_analysis_report.md'")

# 保存模型性能数据
performance_comparison.to_csv('model_performance_comparison.csv', index=False, encoding='utf-8')
print("模型性能对比数据已保存为 'model_performance_comparison.csv'")

# 保存特征重要性
feature_importance.to_csv('feature_importance.csv', index=False, encoding='utf-8')
print("特征重要性数据已保存为 'feature_importance.csv'")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
print(f"生成的文件:")
print(f"  1. exploratory_analysis.png - 探索性数据分析图表")
print(f"  2. correlation_heatmap.png - 特征相关性热力图")
print(f"  3. model_performance.png - 模型性能可视化")
print(f"  4. feature_importance.png - 特征重要性排序")
print(f"  5. bike_analysis_report.md - 详细分析报告")
print(f"  6. model_performance_comparison.csv - 模型性能对比数据")
print(f"  7. feature_importance.csv - 特征重要性数据")
print("=" * 80)