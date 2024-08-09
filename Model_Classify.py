# 划分数据集，训练（分类）模型。
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import time


def xgboost(X_train_split, X_val, y_train_split, y_val):



    train_matrix = xgb.DMatrix(X_train_split, label=y_train_split)  # 测试集

    valid_matrix = xgb.DMatrix(X_val, label=y_val)  # 验证集

    params = {
        'booster': 'gbtree',
        # 'objective': 'binary:logistic',
        # 'objective': 'multi:softprob',  # 或 'multi:softprob'
        'objective': 'multi:softmax',
        'num_class': 49,
        'eval_metric': 'mlogloss',
        'learning_rate': 0.05,
        'gamma': 1,
        'min_child_weight': 1.5,
        'max_depth': 5,
        'alpha': 0.1,
        'lambda': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        # 'eta': 0.04,
        'tree_method': 'exact',
        'seed': 2222,
        # 'nthread': 36,
        # 'silent': True,
        'verbosity': 0,
        # 'scale_pos_weight': 5
    }

    watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

    start = time.time()
    # model = xgb.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200, early_stopping_rounds=200)

    model = xgb.train(params, train_matrix, num_boost_round=1000, evals=watchlist, early_stopping_rounds=200)
    end = time.time()
    print(end - start)

    # val_probs = model.predict(xgb.DMatrix(X_val))  # 预测得到的是概率
    # val_pre_lgb = np.argmax(val_probs, axis=1)  # 从概率中获取最大值的索引，即预测的类别标签

    val_pre_lgb = model.predict(xgb.DMatrix(X_val))

    plot_importance(model, ax=None, height=0.2, xlim=None, ylim=None, title='Feature importance', xlabel='F score',
                    ylabel='Features', importance_type='gain', max_num_features=10, grid=True, show_values=True)

    plt.show()

    # 假设 model 是已经训练好的 XGBoost 模型
    scores = model.get_score(importance_type='gain')  # 使用 'gain' 作为重要性类型

    # 将得分按重要性降序排序
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 打印所有特征及其重要性
    for feature, importance in sorted_scores:
        print(f"{feature}: {importance}")

    # 将预测结果二值化，用于计算准确率、精确率、召回率等
    val_pre_lgb = np.around(val_pre_lgb, 0).astype(int)
    accuracy = accuracy_score(y_val, val_pre_lgb)
    print('xgb 模型的准确率：{}'.format(accuracy))
    precision = precision_score(y_val, val_pre_lgb, average='weighted', zero_division=0)
    print('xgb 模型的精确率：{}'.format(precision))
    recall = recall_score(y_val, val_pre_lgb, average='weighted', zero_division=0)
    print('xgb 模型的召回率：{}'.format(recall))
    # ks = max(abs(fpr - tpr))
    # print('xgb 模型的 ks 值：{}'.format(ks))
    f1 = f1_score(y_val, val_pre_lgb, average='weighted', zero_division=0)
    print('xgb 模型的 F1 值：{}'.format(f1))


def lightgbm(X_train_split, X_val, y_train_split, y_val):

    # 初始化LightGBM模型
    model = LGBMClassifier(
        boosting_type='gbdt',
        objective='multiclass',  # 指定多分类任务
        num_class=49,  # 类别总数
        num_leaves=40,
        max_depth=8,
        learning_rate=0.05,
        feature_fraction=0.9,  # 使用 feature_fraction 而不是 colsample_bytree
        bagging_fraction=0.8,  # 使用 bagging_fraction 而不是 subsample
        bagging_freq=12,  # 使用 bagging_freq 而不是 subsample_freq
        metric='multi_logloss'  # 多分类的损失函数
    )

    # 训练模型
    # model.fit(X_train_split, y_train_split, eval_set=[(X_val, y_val)], eval_metric='multi_logloss',
    #           early_stopping_rounds=50)

    model.fit(
        X_train_split,
        y_train_split,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss'  # 在50轮迭代内如果验证集上的multi_logloss没有改善，则停止训练
    )
    # 预测验证集
    y_pred = model.predict(X_val)

    # 计算准确率
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy}")


import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import KFold, train_test_split


def lightgbm2(X_train_split, X_val, y_train_split, y_val):
    models = {}  # 用于保存每个trial的模型
    def train_model_category(trial, X_train_split, y_train_split, X_val, y_val):
        param_grid = {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "num_class": 49,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01),
            "num_leaves": trial.suggest_int("num_leaves", 30, 60, step=5),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100, step=10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0, step=0.1),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0, step=0.1),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 1, step=0.1),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 1, step=0.1),
            "verbose": -1
        }

        clf = LGBMClassifier(**param_grid)
        clf.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=[LightGBMPruningCallback(trial, "multi_logloss")]
            # callbacks = [LightGBMPruningCallback(trial, "multi_error")]
            # callbacks = [LightGBMPruningCallback(trial, "multi_softmax")]

        )

        # 通过 predict_proba 获得概率，计算 log loss
        # preds_proba = clf.predict_proba(X_val)
        # return log_loss(y_val, preds_proba)  # 返回 log loss 用于优化

        preds_proba = clf.predict_proba(X_val)
        preds = clf.predict(X_val)
        loss = log_loss(y_val, preds_proba)
        accuracy = accuracy_score(y_val, preds)
        precision = precision_score(y_val, preds, average='macro')  # Use 'micro' or 'weighted' based on your requirement
        recall = recall_score(y_val, preds, average='macro')  # Use 'micro' or 'weighted' based on your requirement

        # 将模型存储在外部字典中，而不是在 trial 的 user_attrs 中
        models[trial.number] = clf  # 使用 trial 的编号作为键

        return {"log_loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall}

        # preds = clf.predict(X_val)
        # accuracy = accuracy_score(y_val, preds)
        # return accuracy

    def objective(trial):

        # return train_model_category(trial, X_train_split, y_train_split, X_val, y_val)
        metrics = train_model_category(trial, X_train_split, y_train_split, X_val, y_val)
        trial.set_user_attr("accuracy", metrics["accuracy"])
        trial.set_user_attr("precision", metrics["precision"])
        trial.set_user_attr("recall", metrics["recall"])
        return metrics["log_loss"]  # You can switch to optimizing another metric by returning a different one

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial

    # Output the best trial results
    best_trial = study.best_trial
    # 使用最佳试验的编号从字典中提取模型
    best_model = models[trial.number]

    print(f"Best trial log_loss: {best_trial.values}")
    print(f"Accuracy: {best_trial.user_attrs['accuracy']}")
    print(f"Precision: {best_trial.user_attrs['precision']}")
    print(f"Recall: {best_trial.user_attrs['recall']}")

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # 输出特征重要性
    feature_importances = best_model.feature_importances_
    sorted_idx = feature_importances.argsort()
    print("Feature importances:")
    for idx in sorted_idx[::-1]:  # 输出从最重要到最不重要的特征
        print(f"{X_train_split.columns[idx]}: {feature_importances[idx]}")

    return best_model


    # 假设 feature_importances 来自最佳模型，X_train_split 是原始的训练数据集

    # features = X_train_split.columns
    #
    # # 获取重要性为0的特征索引
    # zero_importance_features = features[feature_importances == 0]
    #
    # # 移除这些特征
    # X_train_reduced = X_train_split.drop(columns=zero_importance_features)
    # X_val_reduced = X_val.drop(columns=zero_importance_features)

    # # 重新训练模型
    # clf_retrained = LGBMClassifier(**best_trial.params)
    # clf_retrained.fit(X_train_reduced, y_train_split,
    #                   eval_set=[(X_val_reduced, y_val)])
    #
    # # 评估新模型
    # preds_retrained = clf_retrained.predict(X_val_reduced)
    # accuracy_retrained = accuracy_score(y_val, preds_retrained)
    # print(f'Retrained Model Accuracy: {accuracy_retrained}')
    # precision_retrained = precision_score(y_val, preds_retrained, average='macro')
    # recall_retrained = recall_score(y_val, preds_retrained, average='macro')
    #
    # print(f'Precision: {precision_retrained}, Recall: {recall_retrained}')


from catboost import CatBoostClassifier, Pool

def catboost(X, Y):
    # x_train = X_train_split
    # x_test = X_val
    # y_train = y_train_split
    # y_test = y_val
    # 设置 CatBoostClassifier 参数
    clf = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        task_type="CPU",
        learning_rate=0.1,
        iterations=300,
        random_seed=2022,
        od_type="Iter",
        depth=7
    )

    # 初始化评估指标
    mean_accuracy = 0
    mean_f1 = 0
    mean_precision = 0
    mean_recall = 0
    n_folds = 3
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2022)

    # 假设 col 是分类特征的索引列表
    col = []  # 请根据实际情况填充

    start = time.time()

    # 开始 K 折交叉验证
    for train_index, test_index in kf.split(X):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        # 训练模型
        clf.fit(x_train, y_train, verbose=300, cat_features=col)

        # 预测
        y_pred = clf.predict(x_test)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        mean_accuracy += accuracy / n_folds
        mean_f1 += f1 / n_folds
        mean_precision += precision / n_folds
        mean_recall += recall / n_folds

        print(f'Validation Accuracy: {accuracy}')
        print(f'Validation F1 Score: {f1}')
        print(f'Validation Precision: {precision}')
        print(f'Validation Recall: {recall}')

    # 输出平均评估指标
    print(f'Mean Validation Accuracy: {mean_accuracy}')
    print(f'Mean Validation F1 Score: {mean_f1}')
    print(f'Mean Validation Precision: {mean_precision}')
    print(f'Mean Validation Recall: {mean_recall}')

    end = time.time()
    print(f'Total Time: {end - start} seconds')

