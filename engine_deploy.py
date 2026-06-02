"""
FactorLab — M2 引擎层(engine.py)
====================================
把 M1 的回测逻辑封装成"输入参数 → 返回结果"的纯函数,供看板调用。

设计要点(看板架构的核心):
  计算与展示分离。看板(app.py)不直接碰原始数据,只调用这里的函数。
  最重的一步(清洗对齐 1258 万行)只做一次,产出 panel.parquet 缓存到磁盘;
  之后用户在看板上调"分组数"等参数时,只在干净面板上重算分组/收益(快),
  不再重读原始 CSV。
"""

import os
import pandas as pd
import numpy as np

# 复用 M1 已经写好、验证过的函数 —— 不重复造轮子
from grouping import assign_groups
from group_returns import compute_group_returns
from metrics import (
    cumulative_returns, summarize, summarize_by_year,
    annualized_return, annualized_vol, sharpe_ratio, max_drawdown,
)

import os as _os
# 部署优先读抽样数据;本地有全量则用全量
_here = _os.path.dirname(_os.path.abspath(__file__))
if _os.path.exists(_os.path.join(_here, "sample_panel.parquet")):
    PANEL_PATH = _os.path.join(_here, "sample_panel.parquet")
else:
    PANEL_PATH = _os.path.join(_here, "panel.parquet")


def load_clean_panel(panel_path=PANEL_PATH):
    """读 M1 清洗好的干净面板(date, code, ret1d, factor)。"""
    return pd.read_parquet(panel_path)


def run_backtest(panel, n_groups=10, start=None, end=None):
    """
    给定干净面板和参数,跑一次分层回测,返回看板需要的所有结果。

    参数:
      n_groups : 分几组(看板上可调 5/10/20)
      start,end: 可选的时间区间筛选(看板上可调)

    返回 dict:
      nav      : 各组累计净值(宽表)
      group_ret: 各组每日收益(宽表)
      stats    : 全期指标表
      by_year  : 分年度收益表
      ls        : 多空组合(组1 - 组N)的指标 dict
    """
    df = panel
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]

    df = assign_groups(df, n_groups=n_groups)
    group_ret = compute_group_returns(df)
    nav = cumulative_returns(group_ret)
    stats = summarize(group_ret)
    by_year = summarize_by_year(group_ret)

    lo, hi = group_ret.columns.min(), group_ret.columns.max()
    ls_series = group_ret[lo] - group_ret[hi]
    ls = {
        "年化收益": annualized_return(ls_series),
        "年化波动": annualized_vol(ls_series),
        "Sharpe": sharpe_ratio(ls_series),
        "最大回撤": max_drawdown(ls_series),
        "nav": (1 + ls_series).cumprod(),
        "low": int(lo), "high": int(hi),
    }
    return {
        "nav": nav, "group_ret": group_ret,
        "stats": stats, "by_year": by_year, "ls": ls,
    }


if __name__ == "__main__":
    # 自测:不靠看板,直接验证引擎能跑
    panel = load_clean_panel()
    res = run_backtest(panel, n_groups=10)
    print("引擎自测通过。各组年化收益:")
    print((res["stats"]["年化收益"] * 100).round(2).to_string())
    print(f"\n多空 Sharpe: {res['ls']['Sharpe']:.3f}")
