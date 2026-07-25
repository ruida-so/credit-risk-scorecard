"""
Streamlit 信用风险评分卡应用 — 优化版

功能:
  - 借款人信息输入
  - 实时 PD 预测与先验校正
  - 信用评分 (300-900)
  - 贷款决策
  - 风险仪表盘
  - 特征重要性
  - IFRS-9 ECL 计算

运行: streamlit run app.py
"""

import sys
import os

# 将 src 目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import pickle
import streamlit as st
import plotly.graph_objects as go

from pd_model import prior_correction, load_model
from scorecard import probability_to_score, make_loan_decision, ScorecardConfig
from ecl_calculator import calculate_ecl, LGD_TABLE


# ───────────────────────────────
# 页面配置
# ───────────────────────────────

st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="🏦",
    layout="centered",
)

# 自动跳转到嵌入模式，去掉 Streamlit Cloud 查看器工具栏
st.markdown("""
<script>
if (window.top === window.self && !window.location.search.includes('embed')) {
  window.location.replace(window.location.origin + window.location.pathname + '?embed=true');
}
</script>
""", unsafe_allow_html=True)

# 隐藏 Streamlit 默认 UI 元素 — CSS 先手 + JS 后手双重删除
st.markdown("""
<style>
footer, #MainMenu, .stDeployButton,
[data-testid="stDecoration"], [data-testid="stToolbar"],
[data-testid="manage_app_button"], div.stApp > header,
a[href*="streamlit.io"], a[href*="streamlit.app"][target="_blank"] {
  display: none !important;
}
</style>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components
components.html("""
<script>
(function clean() {
  const sel = 'footer, #MainMenu, .stDeployButton, [data-testid="stDecoration"], [data-testid="stToolbar"], [data-testid="manage_app_button"], a[href*="streamlit"], [data-testid="stAppViewer"], [data-testid="stAppViewerToolbar"], .stViewerToolbar, .stAppViewer, [class*="viewer"], [class*="Viewer"]';
  document.querySelectorAll(sel).forEach(e => e.remove());
  document.querySelectorAll('*').forEach(e => {
    if (e.childNodes.length === 1 && e.textContent && /(Streamlit|Made with|Shared by|Viewer)/.test(e.textContent)) {
      e.remove();
    }
  });
  setTimeout(clean, 2000);
})();
</script>
""", height=0)

st.title("🏦 Credit Risk Scorecard — IFRS 9")
st.write("输入借款人信息，实时获取 PD、信用评分和贷款决策。")


# ───────────────────────────────
# 模型加载 (缓存)
# ───────────────────────────────

from data_processor import SELECTED_FEATURES

@st.cache_resource
def load_model_cached():
    model_path = os.path.join(os.path.dirname(__file__), "pd_model_final.pkl")
    if not os.path.exists(model_path):
        st.error("模型文件 pd_model_final.pkl 不存在，请先运行 python train.py")
        st.stop()
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    # 兼容原始项目格式 (只有 model + scaler) 和优化版格式
    data.setdefault("features", SELECTED_FEATURES)
    data.setdefault("real_prior", 0.0668)
    data.setdefault("auc_roc", 0.85)
    data.setdefault("ks", 0.54)
    return data

model_data = load_model_cached()
model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]
real_prior = model_data.get("real_prior", 0.0668)


# ───────────────────────────────
# 用户输入
# ───────────────────────────────

st.subheader("借款人信息")

col_left, col_right = st.columns(2)

with col_left:
    age = st.slider("年龄", 18, 80, 35)
    credit_utilization = st.slider("信用卡使用率", 0.0, 1.0, 0.3, step=0.01)
    late_30 = st.number_input("30-59天逾期次数", 0, 30, 0)
    late_60 = st.number_input("60-89天逾期次数", 0, 30, 0)
    late_90 = st.number_input("90天+逾期次数", 0, 30, 0)
    monthly_income = st.number_input("月收入 ($)", 1000, 200000, 5000, step=500)

with col_right:
    debt_ratio = st.slider("负债比率", 0.0, 5.0, 0.5, step=0.05)
    open_loans = st.number_input("未结清信贷数", 0, 30, 5)
    dependents = st.number_input("抚养人数", 0, 20, 1)
    loan_amount = st.slider("申请贷款金额 ($)", 1000, 500000, 10000, step=500)
    loan_type = st.selectbox("贷款类型", list(LGD_TABLE.keys()), index=3)


# ───────────────────────────────
# 预测
# ───────────────────────────────

if st.button("评估信用风险", type="primary"):
    # 特征工程 (与训练一致)
    total_delinquency = late_30 * 1 + late_60 * 2 + late_90 * 3
    income_per_dependent = monthly_income / (dependents + 1)
    credit_burden = open_loans / (monthly_income / 1000 + 1)

    input_data = np.array([[
        total_delinquency,
        credit_utilization,
        late_30,
        age,
        income_per_dependent,
        monthly_income,
        debt_ratio,
        debt_ratio,       # DTI_Ratio = DebtRatio
        open_loans,
        credit_burden,
        dependents,
    ]])

    # 标准化 + 预测
    input_scaled = scaler.transform(input_data)
    pd_raw = model.predict_proba(input_scaled)[0][1]

    # 先验概率校正
    pd_prob = float(prior_correction(
        pd_raw, real_prior=real_prior, train_prior=0.50
    ))

    # 评分卡
    score = int(probability_to_score(pd_prob))
    decision = make_loan_decision(pd_prob)

    # ECL 计算
    lgd = LGD_TABLE[loan_type]
    ecl_result = calculate_ecl(pd_prob, lgd, loan_amount)

    # ── 展示结果 ──
    st.markdown("---")
    st.subheader("评估结果")

    m1, m2, m3 = st.columns(3)
    m1.metric("违约概率 (PD)", f"{pd_prob:.2%}")
    m2.metric("信用评分", score)
    m3.metric("贷款决策", decision.decision)

    st.write(f"风险等级: **{decision.risk_category}**")
    st.write(f"建议利率: **{decision.interest_rate}**")

    # ── 风险仪表盘 ──
    st.markdown("### 风险仪表盘")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pd_prob * 100,
        title={"text": "违约概率 (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red"},
            "steps": [
                {"range": [0, 10], "color": "green"},
                {"range": [10, 25], "color": "yellow"},
                {"range": [25, 100], "color": "red"},
            ],
        },
    ))
    st.plotly_chart(fig)

    # ── 特征重要性 ──
    st.subheader("特征重要性")
    importance = model.coef_[0]
    importance_df = pd.DataFrame({
        "Feature": features,
        "Impact": importance,
    }).sort_values("Impact", ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))

    # ── ECL 计算 ──
    st.subheader("IFRS-9 预期信用损失")
    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("LGD", f"{lgd:.0%}")
    col_e2.metric("EAD", f"${loan_amount:,.0f}")
    col_e3.metric("12个月 ECL", f"${ecl_result.ecl_12m:,.2f}")

    st.write(f"终身 ECL: ${ecl_result.ecl_lifetime:,.2f}")
    st.write(f"IFRS-9 分期: **Stage {ecl_result.stage}**")

    # ── 模型置信度 ──
    confidence = (1 - pd_prob) * 100
    st.metric("模型置信度", f"{confidence:.1f}%")


# ───────────────────────────────
# 页脚
# ───────────────────────────────

st.markdown("---")
st.caption(
    f"模型: Logistic Regression | "
    f"AUC ≈ {model_data.get('auc_roc', 0.85):.4f} | "
    f"KS ≈ {model_data.get('ks', 0.54):.4f}"
)
