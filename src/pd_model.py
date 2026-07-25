"""
PD (Probability of Default) 模型模块

职责:
  - 逻辑回归训练 (Basel 标准模型)
  - SMOTE 类别平衡
  - 先验概率校正 (Prior Probability Correction)
  - 模型评估 (AUC, Gini, KS)
  - 模型持久化

优化点:
  - 校正逻辑独立为纯函数，便于单独测试
  - 评估指标一次性计算并封装为 dataclass
  - 类型标注全覆盖
"""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


@dataclass
class ModelMetrics:
    """模型评估指标"""
    auc_roc: float
    gini: float
    ks_statistic: float
    recall: float
    precision: float
    cv_auc_mean: float
    cv_auc_std: float
    tn: int
    fp: int
    fn: int
    tp: int


def prior_correction(
    p_model: np.ndarray | float,
    real_prior: float = 0.0668,
    train_prior: float = 0.50,
) -> np.ndarray | float:
    """
    先验概率校正

    当模型在 SMOTE 平衡数据 (50/50) 上训练时，
    预测概率会系统性高估违约率。
    此函数将概率校正回真实分布。

    公式:
      P_real = P_model × (π_real / π_train) /
               [P_model × (π_real / π_train) + (1-P_model) × ((1-π_real)/(1-π_train))]

    Args:
        p_model: 模型输出的违约概率
        real_prior: 真实违约率 (默认 6.68%)
        train_prior: 训练集违约率 (SMOTE 后为 50%)
    """
    p = np.asarray(p_model, dtype=float)
    numerator = p * (real_prior / train_prior)
    denominator = numerator + (1 - p) * ((1 - real_prior) / (1 - train_prior))
    corrected = numerator / denominator
    return np.clip(corrected, 0.0001, 0.9999)


def train_pd_model(
    df: pd.DataFrame,
    features: list[str],
    target: str = "SeriousDlqin2yrs",
    test_size: float = 0.2,
    random_state: int = 42,
    smote_k_neighbors: int = 5,
    decision_threshold: float = 0.30,
) -> dict:
    """
    完整的 PD 模型训练流程

    Returns:
        dict 包含: model, scaler, features, X_test, y_test,
        y_pred_proba, y_pred_class, metrics, prior_correction_fn
    """
    X = df[features].copy()
    y = df[target].copy()

    # 分层划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # SMOTE 过采样
    smote = SMOTE(random_state=random_state, k_neighbors=smote_k_neighbors)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    # 逻辑回归
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=random_state,
        class_weight=None,  # SMOTE 已平衡
    )
    model.fit(X_train_scaled, y_train_bal)

    # 预测
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_class = (y_pred_proba >= decision_threshold).astype(int)

    # 评估指标
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    gini = 2 * auc_roc - 1

    # KS 统计量
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    ks_statistic = max(tpr - fpr)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred_class)
    tn, fp, fn, tp = cm.ravel()

    # 交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_test_scaled, y_test, cv=cv, scoring="roc_auc")

    metrics = ModelMetrics(
        auc_roc=auc_roc,
        gini=gini,
        ks_statistic=ks_statistic,
        recall=tp / (tp + fn) if (tp + fn) > 0 else 0,
        precision=tp / (tp + fp) if (tp + fp) > 0 else 0,
        cv_auc_mean=cv_scores.mean(),
        cv_auc_std=cv_scores.std(),
        tn=tn, fp=fp, fn=fn, tp=tp,
    )

    _print_metrics(metrics)

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_proba": y_pred_proba,
        "y_pred_class": y_pred_class,
        "metrics": metrics,
        "real_prior": float(y_test.mean()),
    }


def _print_metrics(m: ModelMetrics) -> None:
    """打印模型评估指标"""
    print("=" * 55)
    print("模型评估指标")
    print("=" * 55)
    print(f"  AUC-ROC:    {m.auc_roc:.4f}  (行业标准 >0.75)")
    print(f"  Gini:       {m.gini:.4f}  (行业标准 >0.60)")
    print(f"  KS 统计量:  {m.ks_statistic:.4f}  (行业标准 >0.30)")
    print(f"  Recall:     {m.recall * 100:.1f}%  (违约捕获率)")
    print(f"  Precision:  {m.precision * 100:.1f}%")
    print(f"  CV AUC:     {m.cv_auc_mean:.4f} (±{m.cv_auc_std:.4f})")
    print(f"  混淆矩阵: TN={m.tn:,} FP={m.fp:,} FN={m.fn:,} TP={m.tp:,}")


def save_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    features: list[str],
    real_prior: float,
    metrics: ModelMetrics,
    filepath: str,
) -> None:
    """保存模型及所有元数据"""
    model_data = {
        "model": model,
        "scaler": scaler,
        "features": features,
        "prior_correct": prior_correction,
        "real_prior": real_prior,
        "auc_roc": metrics.auc_roc,
        "gini": metrics.gini,
        "ks": metrics.ks_statistic,
    }
    with open(filepath, "wb") as f:
        pickle.dump(model_data, f)
    print(f"模型已保存: {filepath}")


def load_model(filepath: str) -> dict:
    """加载模型"""
    with open(filepath, "rb") as f:
        return pickle.load(f)
