"""
信用评分卡模块

职责:
  - 将 PD 概率转换为 300-900 信用评分
  - 评分分档 (Poor / Fair / Good / Very Good / Exceptional)
  - 贷款决策 (批准 / 有条件批准 / 拒绝)

使用 Points to Double Odds (PDO) 方法，
这是 CIBIL、FICO 等信用评分机构的标准方法。
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ScorecardConfig:
    """评分卡参数"""
    pdo: int = 20            # odds 翻倍时分数变化量
    base_score: int = 600    # 基础 odds 对应的分数
    base_odds: int = 14      # 基础 good:bad 比率 (14:1 ≈ 6.68% PD，匹配真实数据)
    score_min: int = 300
    score_max: int = 900


def probability_to_score(
    pd_probability: np.ndarray | float,
    config: ScorecardConfig | None = None,
) -> np.ndarray:
    """
    将违约概率转换为信用评分

    PDO 方法:
      score = offset + factor × ln(odds)
      odds  = (1 - PD) / PD
      factor = PDO / ln(2)
      offset = base_score - factor × ln(base_odds)

    Args:
        pd_probability: 违约概率
        config: 评分卡参数
    """
    if config is None:
        config = ScorecardConfig()

    p = np.asarray(pd_probability, dtype=float)
    p = np.clip(p, 0.0001, 0.9999)

    odds = (1 - p) / p
    factor = config.pdo / np.log(2)
    offset = config.base_score - factor * np.log(config.base_odds)

    score = offset + factor * np.log(odds)
    score = np.clip(score, config.score_min, config.score_max)

    return score.astype(int)


def score_to_band(score: int | np.ndarray) -> str:
    """将分数映射到风险等级"""
    if np.isscalar(score):
        s = int(score)
    else:
        return np.array([score_to_band(s) for s in score])

    if s >= 650:
        return "优秀 (650+)"
    elif s >= 620:
        return "良好 (620-649)"
    elif s >= 580:
        return "中等 (580-619)"
    elif s >= 500:
        return "一般 (500-579)"
    else:
        return "差 (300-499)"


@dataclass
class LoanDecision:
    """贷款决策结果"""
    credit_score: int
    pd_probability: float
    decision: str
    interest_rate: str
    risk_category: str
    expected_loss: float = 0.0
    loan_to_income: float = 0.0
    rejection_reason: str = ""


def make_loan_decision(
    pd_probability: float,
    loan_amount: float = 10000,
    monthly_income: float = 5000,
    config: ScorecardConfig | None = None,
) -> LoanDecision:
    """
    根据违约概率 + 贷款金额做出贷款决策

    两维判断：
      1. 信用评分 (来自 PD)
      2. 预期损失 = PD × 贷款金额；贷款收入比 = 贷款金额 ÷ 年收入

    规则：
      - 预期损失 > $3,000 或 贷款收入比 > 500% → 降一级
      - 预期损失 > $10,000 或 贷款收入比 > 1000% → 直接拒绝
    """
    annual_income = monthly_income * 12
    loan_to_income = loan_amount / max(annual_income, 1)
    expected_loss = pd_probability * loan_amount

    score = int(probability_to_score(pd_probability, config))

    if score >= 650:
        decision = "批准"
        rate = "年利率 10.5%"
        risk = "低风险"
    elif score >= 580:
        decision = "有条件批准"
        rate = "年利率 15.5%"
        risk = "中风险"
    else:
        decision = "拒绝"
        rate = "不适用"
        risk = "高风险"

    # 敞口维度调整
    reason = ""
    if decision != "拒绝":
        if expected_loss > 8000 or loan_to_income > 5.0:
            decision = "拒绝"
            rate = "不适用"
            risk = "高风险"
            reason = f"贷款敞口过高（预期损失 ${expected_loss:,.0f}, 贷款/年收入 {loan_to_income:.0%}）"
        elif expected_loss > 2000 or loan_to_income > 3.0:
            if decision == "批准":
                decision = "有条件批准"
                risk = "中风险"
                reason = f"贷款敞口偏高（预期损失 ${expected_loss:,.0f}, 贷款/年收入 {loan_to_income:.0%}）"
            elif decision == "有条件批准":
                reason = f"贷款敞口偏高（预期损失 ${expected_loss:,.0f}, 贷款/年收入 {loan_to_income:.0%}）"

    return LoanDecision(
        credit_score=score,
        pd_probability=pd_probability,
        decision=decision,
        interest_rate=rate,
        risk_category=risk,
        expected_loss=expected_loss,
        loan_to_income=loan_to_income,
        rejection_reason=reason,
    )
