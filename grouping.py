"""
FactorLab — 模块二:每日分组(quantile grouping)
====================================================
目标:给清洗后的面板,新增一列 group,表示"该 (date, code) 当天属于第几组"。

核心规则(分层回测的灵魂):
  - 必须 **按天独立** 分组(横截面)。每个交易日内,把当天所有股票按因子值
    排序,均分成 n_groups 组。绝不能跨日期一起排序。
  - 分组方向约定:group=1 → 因子值最小,group=n_groups → 因子值最大。
    本数据因子是市值排名,故 group=1 是小市值组,group=10 是大市值组。
    这个约定写死并贯穿全项目,看曲线/写结论时才不会说反。

三个坑的处理:
  1. 按天分组   → groupby('date') 后在每组内部分箱。
  2. 因子并列   → 用 rank(method='first') 打破并列,保证可稳定切分。
  3. 不能整除10 → 用基于排名百分位的切分,余数自然分到靠前的组,各组至多差 1 只。
"""

import pandas as pd
import numpy as np


def assign_groups_one_day(factor_series, n_groups=10):
    """
    给单个交易日的因子序列分组,返回每只股票的组号(1 ~ n_groups)。

    实现思路:
      不直接对原始因子值用 qcut(并列值会让 qcut 报错或分组不均),
      而是先把因子值转成 **组内排名**(rank),再按排名等分。
      rank(method='first') 对并列值按出现顺序给不同名次 —— 彻底消除并列,
      保证后续切分时每组数量尽可能相等(这是处理坑 2 的标准做法)。

    余数处理(坑 3):
      用 ceil(rank / size * n_groups) 把名次映射到 1~n_groups。
      当 size 不能被 n_groups 整除时,靠前的组会多分到 1 只股票,
      各组数量至多相差 1 —— 行为明确、可解释,不是随机的。
    """
    n = len(factor_series)
    if n < n_groups:
        # 当天股票太少(少于组数),无法有意义地分 n 组,整天作废返回 NaN。
        return pd.Series(np.nan, index=factor_series.index)

    # 1) 升序排名:因子值最小 → 名次 1。method='first' 打破并列。
    ranks = factor_series.rank(method="first", ascending=True)
    # 2) 名次 → 组号:1..n 映射到 1..n_groups,向上取整。
    groups = np.ceil(ranks / n * n_groups).astype(int)
    # 3) 浮点边界保险:确保组号落在 [1, n_groups]。
    groups = groups.clip(1, n_groups)
    return groups


def assign_groups(panel, n_groups=10):
    """
    对整张面板按天分组,新增 'group' 列。

    用 groupby('date').transform / apply 在每个交易日内独立调用上面的单日函数。
    这里用 groupby(...)['factor'].transform 的等价写法,保证返回的 group 与
    原 panel 行一一对齐。
    """
    panel = panel.copy()
    panel["group"] = (
        panel.groupby("date")["factor"]
        .transform(lambda s: assign_groups_one_day(s, n_groups))
    )
    # 丢掉当天股票太少、无法分组的行(group 为 NaN)
    before = len(panel)
    panel = panel.dropna(subset=["group"])
    panel["group"] = panel["group"].astype(int)
    dropped = before - len(panel)
    if dropped:
        print(f"      丢弃无法分组的行(当天股票 < {n_groups}):{dropped:,} 行")
    return panel


if __name__ == "__main__":
    import os
    DATA = "/home/claude/data"
    panel = pd.read_parquet(os.path.join(DATA, "panel.parquet"))
    print(f"读入面板:{len(panel):,} 行")

    N = 10
    panel = assign_groups(panel, n_groups=N)
    print(f"\n分组完成,共 {N} 组。")

    # 自检 1:随便挑一天,看各组数量是否尽可能均等
    sample_day = panel["date"].iloc[len(panel) // 2]
    day = panel[panel["date"] == sample_day]
    print(f"\n[自检] {sample_day.date()} 当天共 {len(day)} 只股票,各组数量:")
    print(day["group"].value_counts().sort_index().to_string())

    # 自检 2:确认分组方向 —— group 越大,平均因子值应越大
    print(f"\n[自检] 各组平均因子值(应随组号单调递增):")
    print(panel.groupby("group")["factor"].mean().round(4).to_string())

    panel.to_parquet(os.path.join(DATA, "panel_grouped.parquet"), index=False)
    print(f"\n已保存分组结果 → {os.path.join(DATA, 'panel_grouped.parquet')}")
