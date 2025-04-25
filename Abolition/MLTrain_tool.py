# ML方法：random forest和xgboost

import pandas as pd
import ast
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

# 选择使用的数据集
CHOOSE = 4  # 1:FixLBasewithMoreS 2:FixLBasewithMoreSBase 3:FixLwithMoreS 4:FixLwithMoreSBase
choose = ['FixLBasewithMoreS', 'FixLBasewithMoreSBase', 'FixLwithMoreS', 'FixLwithMoreSBase']

result_file = 'MLtraining_results.csv'

if not os.path.exists(result_file):
    with open(result_file, 'w') as f:
        f.write('Dataset,Model,Acc,Precision,Recall,F1,ConfusionMatrix\n')

# 加载数据集
# 读取csv文件，提取'PPI'和'CATEGORY'字段，返回样本和标签
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    X = df['PPI'].apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist()  # 序列
    y = df['CATEGORY']  # 标签
    encoder = LabelEncoder()  # 将字符串标签转为整数索引
    y = encoder.fit_transform(y)  # 自动识别所有不同类别并转换
    return X, y, encoder


# 评估模型性能
# 计算准确率、精确率、召回率、F1分数和混淆矩阵
def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm


# 训练机器学习模型
def train_ml_model(model_type='RF'):
    # 加载数据
    if CHOOSE == 1:
        X_train, y_train, _ = load_dataset('C:/ETC_proj/dataset_afterProcess/FixLBasewithMoreS/_train.csv')
        X_val, y_val, _ = load_dataset('C:/ETC_proj/dataset_afterProcess/FixLBasewithMoreS/_val.csv')
        X_test, y_test, encoder = load_dataset('C:/ETC_proj/dataset_afterProcess/FixLBasewithMoreS/_test.csv')

    if CHOOSE == 2:
        X_train, y_train, _ = load_dataset('C:/ETC_proj/dataset_afterProcess/FixLBasewithMoreSBase/_train.csv')
        X_val, y_val, _ = load_dataset('C:/ETC_proj/dataset_afterProcess/FixLBasewithMoreSBase/_val.csv')
        X_test, y_test, encoder = load_dataset('C:/ETC_proj/dataset_afterProcess/FixLBasewithMoreSBase/_test.csv')

    if CHOOSE == 3:
        X_train, y_train, _ = load_dataset('C:/ETC_proj/dataset_afterProcess/FixLwithMoreS/_train.csv')
        X_val, y_val, _ = load_dataset('C:/ETC_proj/dataset_afterProcess/FixLwithMoreS/_val.csv')
        X_test, y_test, encoder = load_dataset('C:/ETC_proj/dataset_afterProcess/FixLwithMoreS/_test.csv')

    if CHOOSE == 4:
        X_train, y_train, _ = load_dataset('C:/ETC_proj/dataset_afterProcess/FixLwithMoreSBase/_train.csv')
        X_val, y_val, _ = load_dataset('C:/ETC_proj/dataset_afterProcess/FixLwithMoreSBase/_val.csv')
        X_test, y_test, encoder = load_dataset('C:/ETC_proj/dataset_afterProcess/FixLwithMoreSBase/_test.csv')

    # 模型选择
    if model_type == 'RF':  # random forest  多颗独立树的平均，并行训练，每棵树用不同的子集随机训练
        model = RandomForestClassifier(n_estimators=100, random_state=42)  # n_estimators随机森林中数的数量
    else:  # xgboost  前一棵树影响下一棵树，串行训练
        model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)  # n_estimators树的数量，使用多分类对数损失

    # 模型训练
    # 没有训练轮次，无法利用验证集使用early stopping
    model.fit(X_train, y_train)

    # 验证集评估
    val_pred = model.predict(X_val)
    val_metrics = evaluate_model(y_val, val_pred)
    val_acc, val_prec, val_rec, val_f1, val_cm = val_metrics

    print(f"dataset->{choose[CHOOSE - 1]}:[验证集] Acc: {val_metrics[0]:.4f}, Precision: {val_metrics[1]:.4f}, Recall: {val_metrics[2]:.4f}, F1: {val_metrics[3]:.4f}")
    print(f"Confusion Matrix:\n{val_metrics[4]}")

    # 测试集评估
    test_pred = model.predict(X_test)
    test_metrics = evaluate_model(y_test, test_pred)
    test_acc, test_prec, test_rec, test_f1, test_cm = test_metrics

    print(f"dataset->{choose[CHOOSE - 1]}:[测试集] Acc: {test_metrics[0]:.4f}, Precision: {test_metrics[1]:.4f}, Recall: {test_metrics[2]:.4f}, F1: {test_metrics[3]:.4f}")
    print(f"Confusion Matrix:\n{test_metrics[4]}")

    with open(result_file, 'a') as f:
        f.write(f"{choose[CHOOSE - 1]},{model_type},{test_acc:.4f},{test_prec:.4f},{test_rec:.4f},{test_f1:.4f},{test_cm.tolist()}\n")

if __name__ == "__main__":
    train_ml_model(model_type='XGB')  # 'RF'或'XGB'