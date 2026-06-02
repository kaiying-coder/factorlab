"""
FactorLab — M2 交互看板(app.py)
===================================
用法:streamlit run app.py

页面结构:
  侧边栏  : 分组数、时间区间 —— 用户调参
  主区     : ① 累计净值曲线(可缩放/悬停) ② 全期指标表 ③ 多空组合
            ④ 分年度热力图
技术要点:
  @st.cache_data 缓存:相同参数的回测结果只算一次,再次访问秒出。
  这是看板流畅的关键 —— 计算重(分组+算收益),但缓存后交互轻。
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from engine_deploy import load_clean_panel, run_backtest

st.set_page_config(page_title="FactorLab 因子分层回测", layout="wide")


# ---- 数据与计算(带缓存)----
@st.cache_data(show_spinner="加载数据中...")
def _load():
    return load_clean_panel()

@st.cache_data(show_spinner="回测计算中...")
def _backtest(n_groups, start, end):
    # 缓存键 = 参数组合;同样参数第二次调用直接返回,不重算
    panel = _load()
    return run_backtest(panel, n_groups=n_groups, start=start, end=end)


import os
if not os.path.exists(__import__("engine_deploy").PANEL_PATH):
    st.error("未找到数据文件 sample_panel.parquet。请先在本地运行 make_sample.py 生成,并与代码一起上传。")
    st.stop()
panel = _load()
dmin, dmax = panel["date"].min().date(), panel["date"].max().date()

# ---- 侧边栏:控制器 ----
st.sidebar.title("⚙️ 回测参数")
n_groups = st.sidebar.select_slider("分组数", options=[5, 10, 20], value=10)
date_range = st.sidebar.slider(
    "时间区间", min_value=dmin, max_value=dmax, value=(dmin, dmax),
)
st.sidebar.caption("因子:市值排名分位 (market_cap)\n组1=小市值 ··· 组N=大市值")

# ---- 跑回测 ----
res = _backtest(n_groups, date_range[0], date_range[1])
nav, stats, by_year, ls = res["nav"], res["stats"], res["by_year"], res["ls"]

# ---- 标题 ----
st.title("📊 FactorLab — 因子分层回测看板")
st.caption(f"市值因子 · A股 · {date_range[0]} ~ {date_range[1]} · 分 {n_groups} 组")

# ---- ① 累计净值曲线 ----
st.subheader("① 各组累计净值(对数坐标)")
fig = go.Figure()
cols = sorted(nav.columns)
for i, g in enumerate(cols):
    # 颜色从红(小市值)到蓝(大市值)
    c = px.colors.sample_colorscale("RdYlBu", i / (len(cols) - 1))[0]
    fig.add_trace(go.Scatter(
        x=nav.index, y=nav[g], name=f"组{g}",
        line=dict(color=c, width=1.5),
        hovertemplate="组" + str(g) + "<br>%{x|%Y-%m-%d}<br>净值 %{y:.2f}<extra></extra>",
    ))
fig.update_yaxes(type="log", title="累计净值(起始=1)")
fig.update_layout(height=480, hovermode="x unified",
                  legend=dict(orientation="h"), margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)

# ---- ② 全期指标表 ----
st.subheader("② 全期统计指标")
disp = stats.copy()
disp["年化收益"] = (disp["年化收益"] * 100).round(2)
disp["年化波动"] = (disp["年化波动"] * 100).round(2)
disp["Sharpe"] = disp["Sharpe"].round(3)
disp["最大回撤"] = (disp["最大回撤"] * 100).round(2)
disp.index = [f"组{g}" for g in disp.index]
# 高亮:年化收益最高的组绿色、最低的红色
st.dataframe(
    disp.style.background_gradient(subset=["年化收益", "Sharpe"], cmap="RdYlGn")
        .format({"年化收益": "{:.2f}%", "年化波动": "{:.2f}%",
                 "Sharpe": "{:.3f}", "最大回撤": "{:.2f}%"}),
    use_container_width=True,
)

# ---- ③ 多空组合 ----
st.subheader(f"③ 多空组合(组{ls['low']} 多头 − 组{ls['high']} 空头)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("年化收益", f"{ls['年化收益']*100:.2f}%")
c2.metric("年化波动", f"{ls['年化波动']*100:.2f}%")
c3.metric("Sharpe", f"{ls['Sharpe']:.3f}")
c4.metric("最大回撤", f"{ls['最大回撤']*100:.2f}%")

# ---- ④ 分年度热力图 ----
st.subheader("④ 分年度收益热力图(%)")
hm = (by_year * 100).round(1)
hm.columns = [f"组{c}" for c in hm.columns]
fig2 = px.imshow(
    hm, color_continuous_scale="RdYlGn", zmin=-80, zmax=80,
    aspect="auto", text_auto=True,
    labels=dict(x="组", y="年份", color="收益%"),
)
fig2.update_layout(height=560, margin=dict(t=20))
st.plotly_chart(fig2, use_container_width=True)

st.caption("FactorLab M2 · 计算与展示分离 + 缓存架构")
