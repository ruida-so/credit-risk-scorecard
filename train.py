"""
一键训练脚本

用法:
    python train.py

前提:
    1. 已下载 cs-training.csv 放在项目根目录
    2. 已安装 requirements.txt 中的依赖
"""

import sys
import os

# 将 src 目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_processor import (
    load_data,
    clean_data,
    engineer_features,
    SELECTED_FEATURES,
)
from woe_iv import compute_iv_summary, select_features_by_iv
from pd_model import train_pd_model, save_model
from scorecard import probability_to_score, score_to_band
from ecl_calculator import (
    calculate_portfolio_ecl,
    simulate_loan_portfolio,
)
import numpy as np
import pandas as pd


def main():
    print("=" * 55)
    print("信用风险评分卡 — 模型训练流程")
    print("Basel III + IFRS 9 框架")
    print("=" * 55)

    # ── Step 1: 数据加载 ──
    data_path = "cs-training.csv"
    if not os.path.exists(data_path):
        print(f"错误: 找不到 {data_path}")
        print("请从 Kaggle 下载 Give Me Some Credit 数据集:")
        print("https://www.kaggle.com/datasets/brycecf/give-me-some-credit")
        sys.exit(1)

    df = load_data(data_path)

    # ── Step 2: 数据清洗 ──
    from data_processor import check_data_quality
    check_data_quality(df)
    df = clean_data(df)

    # ── Step 3: 特征工程 ──
    df = engineer_features(df)
    print(f"特征工程完成: {df.shape}")

    # ── Step 4: WoE/IV 特征选择 ──
    all_features = SELECTED_FEATURES + [
        "NumberRealEstateLoansOrLines",
    ]
    iv_df, woe_tables = compute_iv_summary(
        df, all_features, "SeriousDlqin2yrs"
    )
    print("\nIV 排名:")
    print(iv_df.to_string(index=False))

    selected = select_features_by_iv(iv_df, threshold=0.02)
    # 确保使用项目定义的特征集
    features = [f for f in SELECTED_FEATURES if f in selected]
    if not features:
        features = SELECTED_FEATURES
    print(f"最终建模特征: {features}")

    # ── Step 5: PD 模型训练 ──
    result = train_pd_model(df, features, "SeriousDlqin2yrs")

    # ── Step 6: 先验概率校正 ──
    from pd_model import prior_correction
    y_pred_corrected = prior_correction(
        result["y_pred_proba"],
        real_prior=result["real_prior"],
        train_prior=0.50,
    )
    print(f"\n校正后平均 PD: {y_pred_corrected.mean() * 100:.2f}%")
    print(f"实际违约率:   {result['real_prior'] * 100:.2f}%")

    # ── Step 7: 评分卡 ──
    credit_scores = probability_to_score(y_pred_corrected)
    print(f"\n信用评分分布: Min={credit_scores.min()}, "
          f"Max={credit_scores.max()}, Mean={credit_scores.mean():.0f}")

    # ── Step 8: ECL 计算 ──
    monthly_incomes = result["X_test"]["MonthlyIncome"].values
    lgds, eads = simulate_loan_portfolio(
        len(result["X_test"]), monthly_incomes
    )
    ecl_df = calculate_portfolio_ecl(y_pred_corrected, lgds, eads)

    # ── Step 9: 保存模型 ──
    save_model(
        model=result["model"],
        scaler=result["scaler"],
        features=features,
        real_prior=result["real_prior"],
        metrics=result["metrics"],
        filepath="pd_model_final.pkl",
    )

    print("\n" + "=" * 55)
    print("训练完成!")
    print("=" * 55)
    print(f"AUC-ROC:  {result['metrics'].auc_roc:.4f}")
    print(f"Gini:     {result['metrics'].gini:.4f}")
    print(f"KS:       {result['metrics'].ks_statistic:.4f}")
    print("模型文件: pd_model_final.pkl")
    print("\n运行 streamlit run app.py 启动应用")


if __name__ == "__main__":
    main()
