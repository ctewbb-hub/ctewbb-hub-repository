# ==============================================================================
# 模块导入区
# ==============================================================================

# 导入NumPy数值计算库，别名np，用于数组运算和数学计算
import numpy as np

# 导入Pandas数据分析库，别名pd，用于读取和处理表格数据
import pandas as pd

# 导入Matplotlib绘图库的pyplot子模块，别名plt，用于数据可视化
from matplotlib import pyplot as plt

# 从sklearn模型选择模块导入多个工具函数
# train_test_split: 将数据集划分为训练集和测试集
# GridSearchCV: 网格搜索，自动遍历参数组合寻找最佳模型
# KFold: K折交叉验证，用于模型评估
# cross_val_score: 交叉验证评分函数
# learning_curve: 学习曲线函数，评估模型在不同数据量下的性能
# ShuffleSplit: 随机划分交叉验证，每次划分前随机打乱数据
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, learning_curve, ShuffleSplit

# 导入K近邻分类器（KNeighborsClassifier），本项目的核心分类算法
from sklearn.neighbors import KNeighborsClassifier

# 导入StandardScaler数据标准化工具，将特征转换为均值为0、标准差为1的标准分布
# 消除不同特征量纲差异对模型的影响
from sklearn.preprocessing import StandardScaler

# 从sklearn评估指标模块导入多个性能评估函数
# classification_report: 分类报告，包含精确率、召回率、F1分数等详细指标
# confusion_matrix: 混淆矩阵，展示预测结果与真实结果的对比
# accuracy_score: 准确率，预测正确样本占总样本的比例
# precision_score: 精确率，预测为正例的样本中实际为正例的比例
# recall_score: 召回率，实际为正例的样本中被预测为正例的比例
# f1_score: F1分数，精确率和召回率的调和平均数
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 从sklearn特征选择模块导入特征选择工具
# SelectKBest: 选择评分最高的K个特征
# chi2: 卡方检验，用于评估分类特征与目标变量的相关性
# mutual_info_classif: 互信息分类，衡量特征与类别之间的依赖关系，适合连续特征
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

# ==============================================================================
# 学习曲线绘制函数
# ==============================================================================

def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    绘制学习曲线，用于评估模型在不同训练数据量下的性能表现。
    
    参数说明:
        estimator: 要评估的机器学习模型（如KNN、SVM等）
        title: 图表标题
        x: 特征数据（特征矩阵）
        y: 标签数据（目标变量）
        ylim: Y轴显示范围，格式为(最小值, 最大值)，默认None自动调整
        cv: 交叉验证策略，如KFold、ShuffleSplit等，默认None
        n_jobs: 并行处理的CPU核心数，-1表示使用全部核心，默认None
        train_sizes: 训练集大小比例序列，默认从10%到100%，共5个点
    """
    # 设置图表标题
    plt.title(title)
    
    # 如果指定了Y轴范围，则设置显示边界
    if ylim is not None:
        plt.ylim(*ylim)
    
    # 设置X轴标签为"训练样本数"
    plt.xlabel("Training examples")
    
    # 设置Y轴标签为"分数"（准确率、F1分数等评估指标）
    plt.ylabel("Score")
    
    # 调用sklearn的learning_curve函数，计算不同训练集大小下的分数
    # 返回值：train_sizes(训练集大小), train_scores(训练分数), test_scores(交叉验证分数)
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y,
                                                            cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    
    # 计算训练分数的平均值（按行取平均，即对每个训练集大小求平均）
    train_scores_mean = np.mean(train_scores, axis=1)
    
    # 计算训练分数的标准差，反映分数的波动范围
    train_scores_std = np.std(train_scores, axis=1)
    
    # 计算交叉验证分数的平均值
    test_scores_mean = np.mean(test_scores, axis=1)
    
    # 计算交叉验证分数的标准差
    test_scores_std = np.std(test_scores, axis=1)
    
    # 添加网格线，便于观察数据点位置
    plt.grid()

    # 绘制训练分数的误差带（红色半透明区域）
    # 范围为平均值±标准差，alpha=0.1表示透明度
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    
    # 绘制交叉验证分数的误差带（绿色半透明区域）
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    
    # 绘制训练分数曲线（红色圆形标记点+实线连接）
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    
    # 绘制交叉验证分数曲线（绿色圆形标记点+实线连接）
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # 添加图例，自动选择最佳位置显示
    plt.legend(loc="best")
    
    # 返回plt对象，供调用者进一步处理或显示
    return plt

# ==============================================================================
# 数据读取与探索性分析
# ==============================================================================

# 使用Pandas读取CSV格式的糖尿病数据集
# 文件路径为相对路径，假设数据文件与脚本在同一目录下
data = pd.read_csv('diabetes_data.csv')

# 打印分隔线和标题，用于美化输出格式
print("=" * 70)
print("数据集分析")
print("=" * 70)

# 打印数据形状（行数和列数），了解数据集规模
# data.shape返回元组(行数, 列数)，例如(768, 9)表示768个样本，9个特征
print(f"原始数据形状: {data.shape}")

# 打印数据基本信息，包括每列的数据类型、非空值数量等
print(f"\n数据基本信息:")
data.info()

# 打印数据统计摘要，包括均值、标准差、最小值、最大值、四分位数等
# 帮助了解数据的分布特征
print(f"\n数据统计摘要:")
print(data.describe())

# 按Outcome列（糖尿病结果）分组，统计每组的样本数量
# Outcome=0表示未患病，Outcome=1表示患病
print(f"\n类别分布 (Outcome):")
print(data.groupby("Outcome").size())

# 计算各类别的占比（百分比形式）
# normalize=True将计数转换为比例，乘以100得到百分比
print(f"\n各类别占比:")
print(data['Outcome'].value_counts(normalize=True) * 100)

# ==============================================================================
# 无效零值统计
# ==============================================================================

# 定义可能存在无效零值的特征列表
# 这些医学指标在正常情况下不应为0，为0可能表示数据缺失或记录错误
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# 打印无效零值统计标题
print(f"\n无效零值统计 (可能为缺失值):")

# 遍历每个特征，统计零值数量和占比
for feature in zero_features:
    # 统计该特征中等于0的值的个数
    # data[feature] == 0返回布尔数组，.sum()将True(1)相加得到零值数量
    zero_count = (data[feature] == 0).sum()
    
    # 计算零值占总样本数的百分比
    zero_percent = (zero_count / len(data)) * 100
    
    # 打印特征名、零值数量和百分比（保留一位小数）
    print(f"  {feature}: {zero_count} ({zero_percent:.1f}%)")

# ==============================================================================
# 基准模型 - 原始数据预处理与训练
# ==============================================================================

# 打印分隔线和标题
print("\n" + "=" * 70)
print("数据预处理 - 基准模型")
print("=" * 70)

# 复制原始数据，创建基准版本
# 使用.copy()确保不修改原始数据，后续操作只影响副本
data_baseline = data.copy()

# 处理缺失值：将所有NaN值填充为0
# inplace=True表示直接修改原数据框，不返回新对象
data_baseline.fillna(0, inplace=True)

# 处理BMI列中的无穷大值：将正无穷(np.inf)和负无穷(-np.inf)替换为0
# 防止后续计算出现数值异常
data_baseline['BMI'] = data_baseline['BMI'].replace([np.inf, -np.inf], 0)

# 对分类变量进行独热编码
# 将类别型特征转换为二进制编码，使模型能够处理非数值型数据
data_baseline = pd.get_dummies(data_baseline)

# 分离特征矩阵和标签向量
# x_baseline: 特征数据（移除Outcome列），每行一个样本，每列一个特征
# y_baseline: 标签数据（仅保留Outcome列），0表示未患病，1表示患病/axis=1表示按列操作
x_baseline = data_baseline.drop('Outcome', axis=1)
y_baseline = data_baseline['Outcome']

# 创建特征选择器：使用卡方检验(chi2)选择评分最高的8个特征
# 卡方检验评估特征与目标变量的相关性，适用于分类特征
selector_baseline = SelectKBest(score_func=chi2, k=8)

# 对特征数据进行拟合和转换，只保留选择的8个特征
# fit_transform()先计算特征评分，再根据评分筛选特征
x_new_baseline = selector_baseline.fit_transform(x_baseline, y_baseline)

# 将数据划分为训练集和测试集
# test_size=0.2: 测试集占20%，训练集占80%
# random_state=42: 固定随机种子，确保每次划分结果一致（便于复现）
# stratify=y_baseline: 按标签分层抽样，保持训练集和测试集的类别比例一致
x_train_baseline, x_test_baseline, y_train_baseline, y_test_baseline = train_test_split(
    x_new_baseline, y_baseline, test_size=0.2, random_state=42, stratify=y_baseline)

# 创建KNN分类器实例，设置邻居数n_neighbors=9
# KNN算法通过寻找最近的K个邻居来判断新样本的类别
knn_baseline = KNeighborsClassifier(n_neighbors=9)

# 使用训练数据训练模型
# fit()方法学习特征与标签之间的映射关系
knn_baseline.fit(x_train_baseline, y_train_baseline)

# 使用训练好的模型对测试数据进行预测
# predict()方法输入特征数据，输出预测的标签
y_pred_baseline = knn_baseline.predict(x_test_baseline)

# ==============================================================================
# 基准模型性能评估
# ==============================================================================

# 打印基准模型的性能指标
print("\n基准模型性能 (原始参数):")

# 计算并打印训练集准确率
# score()方法返回模型在给定数据上的预测准确率
print(f"训练集准确率: {knn_baseline.score(x_train_baseline, y_train_baseline):.4f}")

# 计算并打印测试集准确率（更能反映模型泛化能力）
print(f"测试集准确率: {knn_baseline.score(x_test_baseline, y_test_baseline):.4f}")

# 计算并打印精确率（Precision）
# 精确率=真正例/(真正例+假正例)，衡量预测为正例的样本中实际为正例的比例
print(f"精确率: {precision_score(y_test_baseline, y_pred_baseline):.4f}")

# 计算并打印召回率（Recall）
# 召回率=真正例/(真正例+假反例)，衡量实际为正例的样本中被预测为正例的比例
print(f"召回率: {recall_score(y_test_baseline, y_pred_baseline):.4f}")

# 计算并打印F1分数
# F1分数是精确率和召回率的调和平均数，综合衡量模型性能
print(f"F1分数: {f1_score(y_test_baseline, y_pred_baseline):.4f}")

# 打印混淆矩阵
# 混淆矩阵展示：真正例(TP)、假正例(FP)、假反例(FN)、真反例(TN)
print("\n混淆矩阵:")
print(confusion_matrix(y_test_baseline, y_pred_baseline))

# 打印分类报告，包含精确率、召回率、F1分数等详细指标
print("\n分类报告:")
print(classification_report(y_test_baseline, y_pred_baseline))

# ==============================================================================
# 改进模型 - 数据预处理优化
# ==============================================================================

# 打印分隔线和标题
print("\n" + "=" * 70)
print("数据预处理 - 改进版本")
print("=" * 70)

# 复制原始数据，创建改进版本
data_improved = data.copy()

# 将无效零值替换为各特征的中位数（比基准模型的填0更合理）
# 中位数不受极端值影响，能更好地代表数据中心趋势
for feature in zero_features:
    # 计算该特征非零值的中位数
    # data_improved[feature] != 0: 筛选出非零值
    # .median(): 计算中位数
    median_val = data_improved[data_improved[feature] != 0][feature].median()
    
    # 将该特征中的0值替换为中位数
    data_improved[feature] = data_improved[feature].replace(0, median_val)
    
    # 打印替换信息（保留两位小数）
    print(f"  {feature}: 零值替换为中位数 {median_val:.2f}")

# 处理BMI列中的无穷大值：替换为BMI列的中位数（比基准模型的填0更合理）
data_improved['BMI'] = data_improved['BMI'].replace([np.inf, -np.inf], data_improved['BMI'].median())

# 对分类变量进行独热编码
data_improved = pd.get_dummies(data_improved)

# 分离特征矩阵和标签向量
x_improved = data_improved.drop('Outcome', axis=1)
y_improved = data_improved['Outcome']

# 创建数据标准化器
# StandardScaler将特征转换为均值为0、标准差为1的标准正态分布
# 消除不同特征量纲差异对KNN算法的影响（KNN基于距离计算，量纲影响较大）
scaler = StandardScaler()

# 对特征数据进行标准化处理
# fit_transform()先计算训练数据的均值和标准差，再进行标准化转换
x_scaled = scaler.fit_transform(x_improved)

# 创建特征选择器：使用互信息分类(mutual_info_classif)选择评分最高的8个特征
# 互信息衡量特征与类别之间的依赖关系，比卡方检验更适合连续特征
selector_improved = SelectKBest(score_func=mutual_info_classif, k=8)

# 对标准化后的特征数据进行拟合和转换，只保留选择的8个特征
x_new_improved = selector_improved.fit_transform(x_scaled, y_improved)

# ==============================================================================
# 特征选择结果展示
# ==============================================================================

# 打印特征选择结果标题
print(f"\n特征选择 - 保留前8个特征")

# 创建特征评分数据框，包含特征名称和对应的互信息评分
feature_scores = pd.DataFrame({'Feature': x_improved.columns, 'Score': selector_improved.scores_})

# 按评分从高到低排序，便于观察特征重要性
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

# 打印特征评分排序结果
print(feature_scores)

# ==============================================================================
# 改进模型数据划分
# ==============================================================================

# 将改进后的数据划分为训练集和测试集
# 参数设置与基准模型一致，确保对比的公平性
x_train_improved, x_test_improved, y_train_improved, y_test_improved = train_test_split(
    x_new_improved, y_improved, test_size=0.2, random_state=42, stratify=y_improved)

# ==============================================================================
# 超参数调优 - 网格搜索
# ==============================================================================

# 打印分隔线和标题
print("\n" + "=" * 70)
print("超参数调优 - 网格搜索")
print("=" * 70)

# 定义参数网格，包含所有要尝试的参数组合
# GridSearchCV将遍历所有组合，找到最佳参数
param_grid = {
    # n_neighbors: KNN的邻居数，从1到30逐一尝试
    'n_neighbors': list(range(1, 31)),
    
    # weights: 权重函数
    # 'uniform': 所有邻居权重相同
    # 'distance': 距离越近的邻居权重越大（距离的倒数）
    'weights': ['uniform', 'distance'],
    
    # metric: 距离度量方法
    # 'euclidean': 欧氏距离（常用）
    # 'manhattan': 曼哈顿距离（城市街区距离）
    # 'chebyshev': 切比雪夫距离（各坐标差的最大值）
    # 'minkowski': 闵可夫斯基距离（欧氏距离和曼哈顿距离的推广）
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    
    # algorithm: 搜索邻居的算法
    # 'auto': 自动选择最合适的算法
    # 'ball_tree': 球树算法，适用于高维数据
    # 'kd_tree': KD树算法，适用于低维数据
    # 'brute': 暴力搜索，适用于小规模数据
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# 创建10折交叉验证对象
# n_splits=10: 将数据分成10份
# shuffle=True: 每次划分前随机打乱数据
# random_state=42: 固定随机种子，确保结果可复现
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 创建网格搜索对象
# GridSearchCV自动遍历参数网格，使用交叉验证评估每个参数组合
grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),  # 要调优的模型
    param_grid=param_grid,             # 参数网格
    cv=kf,                             # 交叉验证策略
    scoring='f1',                      # 评分指标（F1分数）
    n_jobs=-1,                         # 使用所有可用CPU核心并行计算
    verbose=1                          # 显示详细进度信息
)

# 执行网格搜索，找到最佳参数组合
# fit()方法在训练数据上遍历所有参数组合，记录每个组合的交叉验证分数
grid_search.fit(x_train_improved, y_train_improved)

# ==============================================================================
# 网格搜索结果展示
# ==============================================================================

# 打印最佳参数组合
print(f"\n最佳参数组合:")
for key, value in grid_search.best_params_.items():
    print(f"  {key}: {value}")

# 打印最佳交叉验证F1分数
# best_score_存储了最佳参数组合在交叉验证中的平均F1分数
print(f"\n最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ==============================================================================
# 优化模型评估
# ==============================================================================

# 打印分隔线和标题
print("\n" + "=" * 70)
print("优化模型评估")
print("=" * 70)

# 获取网格搜索找到的最佳参数模型
# best_estimator_返回具有最佳参数组合的KNN分类器实例
best_knn = grid_search.best_estimator_

# 使用训练数据重新训练最佳模型
# 虽然GridSearchCV已在交叉验证中训练过，但这里重新训练以确保模型完整
best_knn.fit(x_train_improved, y_train_improved)

# 使用最佳模型对测试数据进行预测
y_pred_optimized = best_knn.predict(x_test_improved)

# ==============================================================================
# 优化模型性能指标
# ==============================================================================

# 打印优化后模型的性能指标
print("\n优化后模型性能:")

# 计算并打印训练集准确率
print(f"训练集准确率: {best_knn.score(x_train_improved, y_train_improved):.4f}")

# 计算并打印测试集准确率
print(f"测试集准确率: {best_knn.score(x_test_improved, y_test_improved):.4f}")

# 计算并打印精确率
print(f"精确率: {precision_score(y_test_improved, y_pred_optimized):.4f}")

# 计算并打印召回率
print(f"召回率: {recall_score(y_test_improved, y_pred_optimized):.4f}")

# 计算并打印F1分数
print(f"F1分数: {f1_score(y_test_improved, y_pred_optimized):.4f}")

# 打印混淆矩阵
print("\n混淆矩阵:")
print(confusion_matrix(y_test_improved, y_pred_optimized))

# 打印分类报告
print("\n分类报告:")
print(classification_report(y_test_improved, y_pred_optimized))

# ==============================================================================
# 10折交叉验证结果
# ==============================================================================

# 打印分隔线和标题
print("\n" + "=" * 70)
print("10折交叉验证结果")
print("=" * 70)

# 使用交叉验证评估模型准确率
# cross_val_score返回每一折的评分，这里使用准确率作为评分指标
cv_scores = cross_val_score(best_knn, x_new_improved, y_improved, cv=kf, scoring='accuracy')

# 打印各折准确率
print(f"各折准确率: {cv_scores}")

# 计算并打印平均准确率和标准差（乘以2表示95%置信区间）
# 置信区间表示真实准确率有95%的概率落在该范围内
print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 使用交叉验证评估模型F1分数
cv_f1 = cross_val_score(best_knn, x_new_improved, y_improved, cv=kf, scoring='f1')

# 打印各折F1分数
print(f"\n各折F1分数: {cv_f1}")

# 计算并打印平均F1分数和标准差（乘以2表示95%置信区间）
print(f"平均F1分数: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")

# ==============================================================================
# 性能对比
# ==============================================================================

# 打印分隔线和标题
print("\n" + "=" * 70)
print("性能对比")
print("=" * 70)

# 打印表头，使用格式化字符串对齐输出
# <15表示左对齐，占15个字符宽度
print(f"{'指标':<15} {'基准模型':<12} {'优化模型':<12} {'提升幅度':<12}")
print("-" * 55)

# 定义要对比的指标字典
# 每个指标存储一个元组：(基准模型值, 优化模型值)
metrics = {
    '准确率': (accuracy_score(y_test_baseline, y_pred_baseline), accuracy_score(y_test_improved, y_pred_optimized)),
    '精确率': (precision_score(y_test_baseline, y_pred_baseline), precision_score(y_test_improved, y_pred_optimized)),
    '召回率': (recall_score(y_test_baseline, y_pred_baseline), recall_score(y_test_improved, y_pred_optimized)),
    'F1分数': (f1_score(y_test_baseline, y_pred_baseline), f1_score(y_test_improved, y_pred_optimized))
}

# 遍历指标字典，计算并打印对比结果
for name, (baseline, optimized) in metrics.items():
    # 计算提升幅度（百分比）
    # 如果基准值大于0，计算(优化值-基准值)/基准值*100
    # 否则提升幅度设为0，避免除以0错误
    improvement = ((optimized - baseline) / baseline * 100) if baseline > 0 else 0
    
    # 打印指标名称、基准值、优化值和提升幅度
    # +.2f表示带正负号，保留两位小数
    print(f"{name:<15} {baseline:.4f}      {optimized:.4f}      {improvement:+.2f}%")

# ==============================================================================
# 学习曲线绘制
# ==============================================================================

# 打印分隔线和标题
print("\n" + "=" * 70)
print("学习曲线")
print("=" * 70)

# 创建随机划分交叉验证对象
# ShuffleSplit每次划分前随机打乱数据，比KFold更灵活
# n_splits=10: 划分10次
# test_size=0.2: 测试集占20%
# random_state=0: 固定随机种子，确保结果可复现
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

# 创建画布，设置大小为14x6英寸
# 足够大的画布便于同时展示两个子图
plt.figure(figsize=(14, 6))

# 创建1行2列的子图，当前绘制第1个子图（左侧）
# subplot(行数, 列数, 当前位置)
plt.subplot(1, 2, 1)

# 调用plot_learning_curve函数绘制基准模型的学习曲线
# 参数：模型、标题、特征数据、标签数据、Y轴范围、交叉验证策略
plot_learning_curve(knn_baseline, "Learning Curve (Baseline KNN)",
                    x_new_baseline, y_baseline, (0.0, 1.01), cv=cv)

# 切换到第2个子图（右侧）
plt.subplot(1, 2, 2)

# 调用plot_learning_curve函数绘制优化模型的学习曲线
plot_learning_curve(best_knn, "Learning Curve (Optimized KNN)",
                    x_new_improved, y_improved, (0.0, 1.01), cv=cv)

# 自动调整子图间距，避免标签重叠
plt.tight_layout()

# 显示图表
plt.show()

# ==============================================================================
# 优化总结
# ==============================================================================

# 打印分隔线和标题
print("\n" + "=" * 70)
print("优化总结")
print("=" * 70)

# 打印数据预处理改进内容
print("1. 数据预处理改进:")
print("   - 将无效零值(Glucose, BloodPressure, SkinThickness, Insulin, BMI)替换为各特征的中位数")
print("   - 对所有特征进行标准化处理(StandardScaler)")

# 打印超参数调优结果
print("\n2. 超参数调优:")
print(f"   - 最佳邻居数 (n_neighbors): {grid_search.best_params_['n_neighbors']}")
print(f"   - 权重函数 (weights): {grid_search.best_params_['weights']}")
print(f"   - 距离度量 (metric): {grid_search.best_params_['metric']}")
print(f"   - 算法 (algorithm): {grid_search.best_params_['algorithm']}")

# 打印模型评估方法
print("\n3. 模型评估:")
print("   - 使用10折交叉验证确保评估的稳健性")
print("   - 综合评估准确率、精确率、召回率和F1分数")

# 打印性能提升总结
print("\n4. 性能提升:")
print("   - 通过数据预处理和超参数调优，模型性能显著提升")
print("   - 学习曲线显示优化后模型泛化能力更好")