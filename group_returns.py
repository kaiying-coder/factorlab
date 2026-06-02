"""
FactorLab — 模块三:计算每组每日收益(portfolio returns)
============================================================
目标:把"日期 × 股票 × 组号"压缩成"日期 × 组"的收益表 —— 每天每组一个收益数字。

两件事:
  A. 次日对齐(消除前视偏差,look-ahead bias)—— 最关键
  B. 组内等权平均

------------------------------------------------------------
A. 为什么要做次日对齐?
   分层回测的因果方向必须是:用 **第 t 日已知的因子** 去赚 **第 t 日之后** 的收益。
   如果用 t 日因子去对应 t 日自己的收益,而 t 日因子(市值)本身又由 t 日涨跌算出,
   就等于用未来信息预测过去 —— 回测虚高,实盘必亏。这是回测最致命的错误。

   我们无法从外部确证原始 ret1d 到底是"当日"还是"次日"收益。
   与其赌,不如在代码里强制规定时间关系:对每只股票,把它的收益序列按时间
   往前挪一格(shift(-1)),让"第 t 行的因子"对上"第 t+1 个交易日的收益"。
   这样无论原始 ret1d 怎么定义,我们用到的都是分组日之后的收益,逻辑上无前视偏差。

   注意:shift 必须在 **每只股票自己的时间序列内部** 做,不能跨股票,
   所以要先 groupby('code') 再 shift。

B. 组内等权 = 简单算术平均。
   一组 N 只股票各占 1/N 权重,组合当日收益 = 这 N 个收益的简单平均。
   pandas 的 .mean() 算的就是简单平均,直接实现"等权"。
"""

import pandas as pd
import numpy as np


def add_forward_return(panel):
    """
    为每只股票构造"次日收益"列 fwd_ret:
      对每个 code,按日期排序后把 ret1d 上移一格 —— 第 t 行得到第 t+1 日的收益。
    每只股票时间序列的最后一天没有次日收益,fwd_ret 为 NaN,后面会被剔除。
    """
    panel = panel.sort_values(["code", "date"]).copy()
    panel["fwd_ret"] = panel.groupby("code")["ret1d"].shift(-1)
    return panel


def compute_group_returns(panel):
    """
    计算每个 (date, group) 的等权组合收益。

    步骤:
      1. 先做次日对齐,得到 fwd_ret。
      2. 丢掉 fwd_ret 缺失的行(每只股票最后一天)。
      3. groupby(['date','group']) 对 fwd_ret 求 mean = 组内等权收益。
      4. 透视成宽表:行=日期,列=组号(1..n),值=该组当日收益。便于后面累计/画图。

    返回:
      group_ret  : 宽表 DataFrame,index=date,columns=[1..n_groups]
    """
    panel = add_forward_return(panel)

    before = len(panel)
    panel = panel.dropna(subset=["fwd_ret"])
    print(f"      次日对齐后丢弃各股最后一日:{before - len(panel):,} 行")

    # 组内等权平均(长表)
    long = (
        panel.groupby(["date", "group"])["fwd_ret"]
        .mean()
        .reset_index()
        .rename(columns={"fwd_ret": "group_ret"})
    )
    # 透视成宽表:行=date,列=group
    group_ret = long.pivot(index="date", columns="group", values="group_ret")
    group_ret = group_ret.sort_index()
    group_ret.columns = [int(c) for c in group_ret.columns]
    return group_ret


if __name__ == "__main__":
    import os
    DATA = "/home/claude/data"
    panel = pd.read_parquet(os.path.join(DATA, "panel_grouped.parquet"))
    print(f"读入分组面板:{len(panel):,} 行")

    group_ret = compute_group_returns(panel)

    print(f"\n每组每日收益表:{group_ret.shape[0]} 个交易日 × {group_ret.shape[1]} 组")
    print(f"区间:{group_ret.index.min().date()} ~ {group_ret.index.max().date()}")

    # 自检 1:每组的全期日均收益。市值因子若有效应呈现单调(大小市值收益有别)
    print("\n[自检] 各组全期日均收益(×10000,即万分之):")
    print((group_ret.mean() * 10000).round(2).to_string())

    # 自检 2:看前 3 天的每组收益,确认是合理的小数(日收益量级)
    print("\n[自检] 前 3 个交易日各组收益预览:")
    print(group_ret.head(3).round(4).to_string())

    group_ret.to_parquet(os.path.join(DATA, "group_returns.parquet"))
    print(f"\n已保存 → {os.path.join(DATA, 'group_returns.parquet')}")
