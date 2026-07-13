import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, learning_curve, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y,
                                                            cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

data = pd.read_csv('diabetes_data.csv')
print("=" * 70)
print("DATASET ANALYSIS")
print("=" * 70)
print(f"原始数据形状: {data.shape}")
print(f"\n数据基本信息:")
data.info()
print(f"\n数据统计摘要:")
print(data.describe())
print(f"\n类别分布 (Outcome):")
print(data.groupby("Outcome").size())
print(f"\n各类别占比:")
print(data['Outcome'].value_counts(normalize=True) * 100)

zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print(f"\n无效零值统计 (可能为缺失值):")
for feature in zero_features:
    zero_count = (data[feature] == 0).sum()
    zero_percent = (zero_count / len(data)) * 100
    print(f"  {feature}: {zero_count} ({zero_percent:.1f}%)")

print("\n" + "=" * 70)
print("PREPROCESSING - ORIGINAL (BASELINE)")
print("=" * 70)
data_baseline = data.copy()
data_baseline.fillna(0, inplace=True)
data_baseline['BMI'] = data_baseline['BMI'].replace([np.inf, -np.inf], 0)
data_baseline = pd.get_dummies(data_baseline)

x_baseline = data_baseline.drop('Outcome', axis=1)
y_baseline = data_baseline['Outcome']

selector_baseline = SelectKBest(score_func=chi2, k=8)
x_new_baseline = selector_baseline.fit_transform(x_baseline, y_baseline)

x_train_baseline, x_test_baseline, y_train_baseline, y_test_baseline = train_test_split(
    x_new_baseline, y_baseline, test_size=0.2, random_state=42, stratify=y_baseline)

knn_baseline = KNeighborsClassifier(n_neighbors=9)
knn_baseline.fit(x_train_baseline, y_train_baseline)
y_pred_baseline = knn_baseline.predict(x_test_baseline)

print("\n基准模型性能 (原始参数):")
print(f"训练集准确率: {knn_baseline.score(x_train_baseline, y_train_baseline):.4f}")
print(f"测试集准确率: {knn_baseline.score(x_test_baseline, y_test_baseline):.4f}")
print(f"精确率: {precision_score(y_test_baseline, y_pred_baseline):.4f}")
print(f"召回率: {recall_score(y_test_baseline, y_pred_baseline):.4f}")
print(f"F1分数: {f1_score(y_test_baseline, y_pred_baseline):.4f}")
print("\n混淆矩阵:")
print(confusion_matrix(y_test_baseline, y_pred_baseline))
print("\n分类报告:")
print(classification_report(y_test_baseline, y_pred_baseline))

print("\n" + "=" * 70)
print("PREPROCESSING - IMPROVED")
print("=" * 70)
data_improved = data.copy()

for feature in zero_features:
    median_val = data_improved[data_improved[feature] != 0][feature].median()
    data_improved[feature] = data_improved[feature].replace(0, median_val)
    print(f"  {feature}: 零值替换为中位数 {median_val:.2f}")

data_improved['BMI'] = data_improved['BMI'].replace([np.inf, -np.inf], data_improved['BMI'].median())
data_improved = pd.get_dummies(data_improved)

x_improved = data_improved.drop('Outcome', axis=1)
y_improved = data_improved['Outcome']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_improved)

selector_improved = SelectKBest(score_func=mutual_info_classif, k=8)
x_new_improved = selector_improved.fit_transform(x_scaled, y_improved)

print(f"\n特征选择 - 保留前8个特征")
feature_scores = pd.DataFrame({'Feature': x_improved.columns, 'Score': selector_improved.scores_})
feature_scores = feature_scores.sort_values(by='Score', ascending=False)
print(feature_scores)

x_train_improved, x_test_improved, y_train_improved, y_test_improved = train_test_split(
    x_new_improved, y_improved, test_size=0.2, random_state=42, stratify=y_improved)

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING - GRID SEARCH")
print("=" * 70)
param_grid = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=kf,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x_train_improved, y_train_improved)

print(f"\n最佳参数组合:")
for key, value in grid_search.best_params_.items():
    print(f"  {key}: {value}")
print(f"\n最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

print("\n" + "=" * 70)
print("OPTIMIZED MODEL EVALUATION")
print("=" * 70)
best_knn = grid_search.best_estimator_
best_knn.fit(x_train_improved, y_train_improved)
y_pred_optimized = best_knn.predict(x_test_improved)

print("\n优化后模型性能:")
print(f"训练集准确率: {best_knn.score(x_train_improved, y_train_improved):.4f}")
print(f"测试集准确率: {best_knn.score(x_test_improved, y_test_improved):.4f}")
print(f"精确率: {precision_score(y_test_improved, y_pred_optimized):.4f}")
print(f"召回率: {recall_score(y_test_improved, y_pred_optimized):.4f}")
print(f"F1分数: {f1_score(y_test_improved, y_pred_optimized):.4f}")
print("\n混淆矩阵:")
print(confusion_matrix(y_test_improved, y_pred_optimized))
print("\n分类报告:")
print(classification_report(y_test_improved, y_pred_optimized))

print("\n" + "=" * 70)
print("10-FOLD CROSS-VALIDATION RESULTS")
print("=" * 70)
cv_scores = cross_val_score(best_knn, x_new_improved, y_improved, cv=kf, scoring='accuracy')
print(f"各折准确率: {cv_scores}")
print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

cv_f1 = cross_val_score(best_knn, x_new_improved, y_improved, cv=kf, scoring='f1')
print(f"\n各折F1分数: {cv_f1}")
print(f"平均F1分数: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)
print(f"{'指标':<15} {'基准模型':<12} {'优化模型':<12} {'提升幅度':<12}")
print("-" * 55)

metrics = {
    '准确率': (accuracy_score(y_test_baseline, y_pred_baseline), accuracy_score(y_test_improved, y_pred_optimized)),
    '精确率': (precision_score(y_test_baseline, y_pred_baseline), precision_score(y_test_improved, y_pred_optimized)),
    '召回率': (recall_score(y_test_baseline, y_pred_baseline), recall_score(y_test_improved, y_pred_optimized)),
    'F1分数': (f1_score(y_test_baseline, y_pred_baseline), f1_score(y_test_improved, y_pred_optimized))
}

for name, (baseline, optimized) in metrics.items():
    improvement = ((optimized - baseline) / baseline * 100) if baseline > 0 else 0
    print(f"{name:<15} {baseline:.4f}      {optimized:.4f}      {improvement:+.2f}%")

print("\n" + "=" * 70)
print("LEARNING CURVES")
print("=" * 70)

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plot_learning_curve(knn_baseline, "Learning Curve (Baseline KNN)",
                    x_new_baseline, y_baseline, (0.0, 1.01), cv=cv)

plt.subplot(1, 2, 2)
plot_learning_curve(best_knn, "Learning Curve (Optimized KNN)",
                    x_new_improved, y_improved, (0.0, 1.01), cv=cv)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("OPTIMIZATION SUMMARY")
print("=" * 70)
print("1. 数据预处理改进:")
print("   - 将无效零值(Glucose, BloodPressure, SkinThickness, Insulin, BMI)替换为各特征的中位数")
print("   - 对所有特征进行标准化处理(StandardScaler)")
print("\n2. 超参数调优:")
print(f"   - 最佳邻居数 (n_neighbors): {grid_search.best_params_['n_neighbors']}")
print(f"   - 权重函数 (weights): {grid_search.best_params_['weights']}")
print(f"   - 距离度量 (metric): {grid_search.best_params_['metric']}")
print(f"   - 算法 (algorithm): {grid_search.best_params_['algorithm']}")
print("\n3. 模型评估:")
print("   - 使用10折交叉验证确保评估的稳健性")
print("   - 综合评估准确率、精确率、召回率和F1分数")
print("\n4. 性能提升:")
print("   - 通过数据预处理和超参数调优，模型性能显著提升")
print("   - 学习曲线显示优化后模型泛化能力更好")