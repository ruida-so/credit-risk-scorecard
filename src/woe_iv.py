"""
WoE (Weight of Evidence) & IV (Information Value) 模块

职责:
  - 计算各特征的 WoE 和 IV
  - 特征选择 (基于 IV 阈值)
  - WoE 可视化

WoE = ln(好客户分布 / 坏客户分布)
IV  = Σ (好客户分布 - 坏客户分布) × WoE

IV 判断标准:
  < 0.02  : 无预测力
  0.02-0.1: 弱
  0.1-0.3 : 中等
  0.3-0.5 : 强
  > 0.5   : 很强 (需警惕过拟合)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """
    计算单个特征的 WoE 和 IV

    Args:
        df: 数据集
        feature: 特征列名
        target: 目标列名 (0=好, 1=坏)
        n_bins: 分箱数

    Returns:
        (woe_table, iv_total)
        woe_table: 各箱的 WoE、违约率等
        iv_total: 该特征的总 IV 值
    """
    temp = df[[feature, target]].copy()

    # 连续变量分箱: 优先等频分箱，失败则等距分箱
    try:
        temp["bin"] = pd.qcut(temp[feature], q=n_bins, duplicates="drop")
    except Exception:
        temp["bin"] = pd.cut(temp[feature], bins=n_bins, duplicates="drop")

    grouped = temp.groupby("bin", observed=True)[target].agg(
        bad="sum",
        good=lambda x: (x == 0).sum(),
        total="count",
    ).reset_index()

    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()

    # 分布占比
    grouped["dist_bad"] = grouped["bad"] / total_bad
    grouped["dist_good"] = grouped["good"] / total_good

    # 避免 log(0)
    grouped["dist_bad"] = grouped["dist_bad"].replace(0, 0.0001)
    grouped["dist_good"] = grouped["dist_good"].replace(0, 0.0001)

    # WoE
    grouped["WoE"] = np.log(grouped["dist_good"] / grouped["dist_bad"])

    # IV
    grouped["IV_bin"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["WoE"]
    iv_total = grouped["IV_bin"].sum()

    # 违约率
    grouped["default_rate"] = grouped["bad"] / grouped["total"] * 100

    return grouped, iv_total


def iv_label(iv: float) -> str:
    """根据 IV 值判断预测力等级"""
    if iv < 0.02:
        return "无预测力"
    elif iv < 0.1:
        return "弱"
    elif iv < 0.3:
        return "中等"
    elif iv < 0.5:
        return "强"
    else:
        return "很强"


def compute_iv_summary(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[pd.DataFrame, dict]:
    """
    批量计算多个特征的 IV

    Returns:
        iv_df: 排序后的 IV 汇总表
        woe_tables: {feature: woe_table} 字典
    """
    iv_results = []
    woe_tables: dict = {}

    for feature in features:
        try:
            woe_table, iv = calculate_woe_iv(df, feature, target)
            iv_results.append({"Feature": feature, "IV": round(iv, 4)})
            woe_tables[feature] = woe_table
        except Exception as e:
            print(f"  {feature} 计算失败: {e}")

    iv_df = pd.DataFrame(iv_results).sort_values("IV", ascending=False).reset_index(drop=True)
    iv_df["Predictive_Power"] = iv_df["IV"].apply(iv_label)

    return iv_df, woe_tables


def select_features_by_iv(
    iv_df: pd.DataFrame,
    threshold: float = 0.02,
) -> list[str]:
    """根据 IV 阈值筛选特征"""
    selected = iv_df[iv_df["IV"] >= threshold]["Feature"].tolist()
    print(f"特征选择: {len(selected)} 个特征通过 IV >= {threshold}")
    return selected
