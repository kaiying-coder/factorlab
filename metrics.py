"""
FactorLab — 模块四:累计收益 + 统计指标(metrics)
====================================================
输入:group_returns.parquet —— 行=交易日,列=组号(1..10),值=该组当日等权收益。
产出:
  A. 累计收益曲线(全期 + 分年度)—— 看方向:因子有没有用、稳不稳
  B. 统计指标(年化收益、年化波动、Sharpe、最大回撤)—— 看质量:值不值得用

关键细节:
  1. 累计用 **累乘(复利)** 不是累加:(1+r).cumprod() - 1。跨年后差异巨大。
  2. 年化用几何方式:年化收益 = (1+日均)^252 - 1;年化波动 = 日波动 × sqrt(252)。
     252 = A 股一年约定俗成的交易日数。
  3. Sharpe 这里用简化版 = 年化收益 / 年化波动(假设无风险利率为 0)。
     严格定义应减去无风险利率再除以波动;实务里比较因子时常用简化版,够用且可比。
  4. 最大回撤:净值序列从历史最高点回落的最大幅度,取负数表示亏损。
"""

import pandas as pd
import numpy as np

TRADING_DAYS = 252  # A 股年化约定的交易日数


def cumulative_returns(group_ret):
    """
    累计收益曲线(累乘/复利)。
    返回净值曲线:第一天起始为 (1+r1),逐日复利。减 1 即累计收益率。
    """
    return (1 + group_ret).cumprod()


def annualized_return(daily_ret):
    """几何年化收益:把日收益序列的平均复利水平换算到一年。"""
    n = daily_ret.count()
    total_growth = (1 + daily_ret).prod()       # 全期总增长倍数
    return total_growth ** (TRADING_DAYS / n) - 1


def annualized_vol(daily_ret):
    """年化波动率:日收益标准差 × sqrt(252)。"""
    return daily_ret.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_ret):
    """简化 Sharpe = 年化收益 / 年化波动(无风险利率设为 0)。"""
    vol = annualized_vol(daily_ret)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return annualized_return(daily_ret) / vol


def max_drawdown(daily_ret):
    """
    最大回撤:净值从历史最高点回落的最大幅度(返回负数)。
    做法:算净值曲线 → 逐日记录历史最高(running max)→ 当前净值相对最高的跌幅
    → 取最小值(最惨的那次)。
    """
    nav = (1 + daily_ret).cumprod()
    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    return drawdown.min()


def summarize(group_ret):
    """对每一组算 4 个指标,汇总成一张表(行=组,列=指标)。"""
    rows = {}
    for g in group_ret.columns:
        r = group_ret[g].dropna()
        rows[g] = {
            "年化收益": annualized_return(r),
            "年化波动": annualized_vol(r),
            "Sharpe": sharpe_ratio(r),
            "最大回撤": max_drawdown(r),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "group"
    return table


def summarize_by_year(group_ret):
    """
    分年度统计每组的年化收益 —— 看因子稳不稳(是否每年都有效)。
    返回:行=年份,列=组号,值=该组该年的(几何)收益。
    """
    out = {}
    for year, sub in group_ret.groupby(group_ret.index.year):
        out[year] = {g: (1 + sub[g].dropna()).prod() - 1 for g in group_ret.columns}
    table = pd.DataFrame(out).T
    table.index.name = "year"
    return table


if __name__ == "__main__":
    import os
    DATA = "/home/claude/data"
    group_ret = pd.read_parquet(os.path.join(DATA, "group_returns.parquet"))
    group_ret.columns = [int(c) for c in group_ret.columns]
    group_ret = group_ret.sort_index()
    print(f"读入每组每日收益:{group_ret.shape[0]} 天 × {group_ret.shape[1]} 组")

    # 产出 A:累计净值
    nav = cumulative_returns(group_ret)
    nav.to_parquet(os.path.join(DATA, "nav.parquet"))

    # 产出 B:全期指标
    stats = summarize(group_ret)
    print("\n=== 全期统计指标 ===")
    disp = stats.copy()
    disp["年化收益"] = (disp["年化收益"] * 100).round(2).astype(str) + "%"
    disp["年化波动"] = (disp["年化波动"] * 100).round(2).astype(str) + "%"
    disp["Sharpe"] = disp["Sharpe"].round(3)
    disp["最大回撤"] = (disp["最大回撤"] * 100).round(2).astype(str) + "%"
    print(disp.to_string())

    # 多空组合:组1(小市值)多头 - 组10(大市值)空头,看因子纯收益
    long_short = group_ret[1] - group_ret[group_ret.columns.max()]
    print("\n=== 多空组合(组1 - 组10)===")
    print(f"年化收益:{annualized_return(long_short)*100:.2f}%  "
          f"年化波动:{annualized_vol(long_short)*100:.2f}%  "
          f"Sharpe:{sharpe_ratio(long_short):.3f}  "
          f"最大回撤:{max_drawdown(long_short)*100:.2f}%")

    # 分年度收益
    by_year = summarize_by_year(group_ret)
    by_year.to_parquet(os.path.join(DATA, "by_year.parquet"))
    print("\n=== 分年度收益(每组,%)— 看稳定性 ===")
    print((by_year * 100).round(1).to_string())

    stats.to_parquet(os.path.join(DATA, "stats.parquet"))
    print(f"\n已保存 nav / stats / by_year → {DATA}")
