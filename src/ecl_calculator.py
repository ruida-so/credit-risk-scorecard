"""
IFRS-9 预期信用损失 (ECL) 计算模块

ECL = PD × LGD × EAD

  PD  (Probability of Default)  : 违约概率
  LGD (Loss Given Default)       : 违约损失率
  EAD (Exposure at Default)      : 违约风险敞口

IFRS-9 分期:
  Stage 1: 低风险 — 12个月 ECL
  Stage 2: 中风险 — 终身 ECL
  Stage 3: 高风险 (已违约) — 终身 ECL
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


# 按贷款类型的 LGD 参考值 (行业标准)
LGD_TABLE = {
    "Home Loan": 0.25,       # 有房产抵押
    "Car Loan": 0.35,        # 有车辆抵押
    "Gold Loan": 0.15,       # 有黄金抵押
    "Personal Loan": 0.75,   # 无抵押
    "Credit Card": 0.85,     # 无抵押
    "Education Loan": 0.60,  # 无抵押
    "Business Loan": 0.45,  # 部分抵押
}


@dataclass
class ECLResult:
    """单笔贷款的 ECL 计算结果"""
    pd_prob: float
    lgd: float
    ead: float
    ecl_12m: float       # 12个月 ECL
    ecl_lifetime: float  # 终身 ECL
    stage: int           # IFRS-9 分期 (1/2/3)


def determine_ifrs9_stage(pd_prob: float) -> int:
    """
    根据 PD 确定 IFRS-9 分期

    Stage 1: PD < 10%   (低风险, 12个月 ECL)
    Stage 2: 10%-50%    (中风险, 终身 ECL)
    Stage 3: PD > 50%   (高风险/已违约, 终身 ECL)
    """
    if pd_prob < 0.10:
        return 1
    elif pd_prob < 0.50:
        return 2
    else:
        return 3


def calculate_ecl(
    pd_prob: float,
    lgd: float,
    ead: float,
    lifetime_factor: float = 3.0,
) -> ECLResult:
    """
    计算单笔贷款的预期信用损失

    Args:
        pd_prob: 违约概率
        lgd: 违约损失率 (0-1)
        ead: 违约风险敞口 (金额)
        lifetime_factor: 终身 ECL 相对 12 个月 ECL 的倍数
    """
    ecl_12m = pd_prob * lgd * ead
    ecl_lifetime = ecl_12m * lifetime_factor
    stage = determine_ifrs9_stage(pd_prob)

    return ECLResult(
        pd_prob=pd_prob,
        lgd=lgd,
        ead=ead,
        ecl_12m=ecl_12m,
        ecl_lifetime=ecl_lifetime,
        stage=stage,
    )


def calculate_portfolio_ecl(
    pd_probs: np.ndarray,
    lgds: np.ndarray,
    eads: np.ndarray,
    lifetime_factor: float = 3.0,
) -> pd.DataFrame:
    """
    批量计算贷款组合的 ECL

    Returns:
        DataFrame: 每笔贷款的 ECL 明细 + 分期统计
    """
    results = []
    for pd_p, lgd, ead in zip(pd_probs, lgds, eads):
        r = calculate_ecl(pd_p, lgd, ead, lifetime_factor)
        results.append({
            "PD": pd_p,
            "LGD": lgd,
            "EAD": ead,
            "ECL_12M": r.ecl_12m,
            "ECL_Lifetime": r.ecl_lifetime,
            "Stage": r.stage,
        })

    df = pd.DataFrame(results)

    # 组合汇总
    print("=" * 55)
    print("IFRS-9 组合 ECL 汇总")
    print("=" * 55)
    print(f"  总敞口 (EAD):       ${df['EAD'].sum():,.0f}")
    print(f"  12个月 ECL:         ${df['ECL_12M'].sum():,.0f}")
    print(f"  终身 ECL:           ${df['ECL_Lifetime'].sum():,.0f}")

    for stage in [1, 2, 3]:
        subset = df[df["Stage"] == stage]
        if len(subset) > 0:
            print(f"  Stage {stage}: {len(subset):,} 笔, "
                  f"ECL=${subset['ECL_Lifetime'].sum():,.0f} "
                  f"({len(subset) / len(df) * 100:.1f}%)")

    return df


def simulate_loan_portfolio(
    n_loans: int,
    monthly_incomes: np.ndarray,
    lgd_table: dict = LGD_TABLE,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    模拟贷款组合的 LGD 和 EAD

    Args:
        n_loans: 贷款数量
        monthly_incomes: 月收入数组 (用于推算贷款金额)
        lgd_table: 贷款类型 -> LGD 映射
        seed: 随机种子

    Returns:
        (lgd_values, ead_values)
    """
    rng = np.random.default_rng(seed)

    loan_types = list(lgd_table.keys())
    probs = [0.30, 0.20, 0.05, 0.25, 0.10, 0.05, 0.05]

    sampled_types = rng.choice(loan_types, size=n_loans, p=probs)
    lgd_values = np.array([lgd_table[lt] for lt in sampled_types])

    # 贷款金额 ≈ 24 个月收入
    loan_amounts = monthly_incomes * 24

    # 已还款比例 10%-90%
    pct_repaid = rng.uniform(0.10, 0.90, n_loans)
    ead_values = loan_amounts * (1 - pct_repaid)
    ead_values = np.clip(ead_values, 5000, 500000)

    return lgd_values, ead_values
