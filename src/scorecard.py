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


def make_loan_decision(
    pd_probability: float,
    config: ScorecardConfig | None = None,
) -> LoanDecision:
    """
    根据违约概率做出贷款决策

    决策规则:
      score >= 720: 批准 (低风险)
      600-719:      有条件批准 (中风险)
      < 600:        拒绝 (高风险)
    """
    score = int(probability_to_score(pd_probability, config))

    if score >= 650:
        return LoanDecision(
            credit_score=score,
            pd_probability=pd_probability,
            decision="批准",
            interest_rate="年利率 10.5%",
            risk_category="低风险",
        )
    elif score >= 580:
        return LoanDecision(
            credit_score=score,
            pd_probability=pd_probability,
            decision="有条件批准",
            interest_rate="年利率 15.5%",
            risk_category="中风险",
        )
    else:
        return LoanDecision(
            credit_score=score,
            pd_probability=pd_probability,
            decision="拒绝",
            interest_rate="不适用",
            risk_category="高风险",
        )
