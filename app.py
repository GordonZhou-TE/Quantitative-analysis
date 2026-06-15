import os
import time
from datetime import date

import akshare as ak
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# =========================
# 页面基础配置
# =========================

st.set_page_config(
    page_title="银行ETF天弘 SH515290 量化回测看板",
    layout="wide"
)

UP_COLOR = "#e60000"      # 红涨
DOWN_COLOR = "#00a000"    # 绿跌

CACHE_FILE = "SH515290_daily_cache.csv"

MA_WINDOWS = [5, 10, 20, 30, 60, 120, 250, 300]

MA_COLORS = {
    5: "#f39c12",
    10: "#3498db",
    20: "#9b59b6",
    30: "#1abc9c",
    60: "#34495e",
    120: "#e67e22",
    250: "#7f8c8d",
    300: "#2c3e50"
}


# =========================
# 工具函数
# =========================

def normalize_symbol(symbol: str) -> str:
    return "".join([c for c in symbol if c.isdigit()])


def get_strategy_description(
    strategy_name: str,
    buy_pct_input: float,
    sell_pct_input: float,
    overbought: float,
    oversold: float
) -> str:
    """
    返回当前策略描述，用于展示在“当前策略”下方
    """
    if strategy_name == "MACD金叉死叉策略":
        return (
            f"策略描述：当 DIF 由下向上穿越 DEA 形成 MACD 金叉时，"
            f"按当日开盘价使用当前可用现金的 {buy_pct_input:.0f}% 买入；"
            f"当 DIF 由上向下穿越 DEA 形成 MACD 死叉时，"
            f"按当日收盘价卖出当前持仓份额的 {sell_pct_input:.0f}%。"
            f"若到达回测结束日期仍有持仓，则按结束日收盘价全部卖出。"
        )

    if strategy_name == "KDJ策略":
        return (
            f"策略描述：当 K、D 同时处于超卖区，即 K <= {oversold} 且 D <= {oversold}，"
            f"并且 K 由下向上穿越 D 形成 KDJ 金叉时，"
            f"按当日开盘价使用当前可用现金的 {buy_pct_input:.0f}% 买入；"
            f"当 K、D 同时处于超买区，即 K >= {overbought} 且 D >= {overbought}，"
            f"并且 K 由上向下穿越 D 形成 KDJ 死叉时，"
            f"按当日收盘价卖出当前持仓份额的 {sell_pct_input:.0f}%。"
            f"若到达回测结束日期仍有持仓，则按结束日收盘价全部卖出。"
        )

    return "策略描述：暂无。"


def standardize_etf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",

        "Date": "date",
        "Open": "open",
        "Close": "close",
        "High": "high",
        "Low": "low",
        "Volume": "volume",
        "Amount": "amount",
        "Pct_Change": "pct_change",
        "Change": "change",
        "Turnover": "turnover"
    }

    df = df.rename(columns=rename_map)

    required_cols = ["date", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"行情数据缺少必要字段：{missing_cols}，当前字段：{list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "open", "close", "high", "low", "volume",
        "amount", "amplitude", "pct_change", "change", "turnover"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def read_local_cache() -> pd.DataFrame:
    if not os.path.exists(CACHE_FILE):
        return pd.DataFrame()

    try:
        df = pd.read_csv(CACHE_FILE, encoding="utf-8-sig")
        return standardize_etf_df(df)
    except Exception:
        return pd.DataFrame()


def save_local_cache(df: pd.DataFrame):
    try:
        if df is not None and not df.empty:
            df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")
    except Exception:
        pass


@st.cache_data(ttl=3600)
def load_etf_daily_data_online(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str
) -> pd.DataFrame:
    symbol_code = normalize_symbol(symbol)
    last_error = None

    for i in range(3):
        try:
            df = ak.fund_etf_hist_em(
                symbol=symbol_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )

            df = standardize_etf_df(df)

            if not df.empty:
                return df

        except Exception as e:
            last_error = e
            time.sleep(2 + i * 2)

    raise RuntimeError(
        f"在线行情数据获取失败。可能是网络、接口或 akshare 数据源问题。原始错误：{last_error}"
    )


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    return standardize_etf_df(df)


# =========================
# 指标计算
# =========================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # MA
    for window in MA_WINDOWS:
        df[f"MA{window}"] = df["close"].rolling(window=window, min_periods=1).mean()

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])

    df["MACD_GOLDEN"] = (
        (df["DIF"] > df["DEA"]) &
        (df["DIF"].shift(1) <= df["DEA"].shift(1))
    )

    df["MACD_DEATH"] = (
        (df["DIF"] < df["DEA"]) &
        (df["DIF"].shift(1) >= df["DEA"].shift(1))
    )

    # KDJ
    n = 9
    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_n = df["high"].rolling(window=n, min_periods=1).max()

    rsv = (df["close"] - low_n) / (high_n - low_n) * 100
    rsv = rsv.replace([np.inf, -np.inf], np.nan).fillna(50)

    df["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    df["D"] = df["K"].ewm(alpha=1 / 3, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]

    df["KDJ_GOLDEN"] = (
        (df["K"] > df["D"]) &
        (df["K"].shift(1) <= df["D"].shift(1))
    )

    df["KDJ_DEATH"] = (
        (df["K"] < df["D"]) &
        (df["K"].shift(1) >= df["D"].shift(1))
    )

    return df


# =========================
# 图表绘制
# =========================

def plot_dashboard(
    df: pd.DataFrame,
    trade_df: pd.DataFrame = None,
    overbought: float = 80,
    oversold: float = 20
):
    df = df.copy()

    volume_colors = np.where(df["close"] >= df["open"], UP_COLOR, DOWN_COLOR)
    macd_colors = np.where(df["MACD"] >= 0, UP_COLOR, DOWN_COLOR)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.15, 0.2, 0.2],
        subplot_titles=[
            "K线与均线",
            "成交量",
            "MACD",
            "KDJ"
        ]
    )

    # Candlestick 兼容旧版 Plotly：不用 hovertemplate，改用 hovertext + hoverinfo
    k_hover_text = [
        (
            f"日期：{r['date'].strftime('%Y/%m/%d')}<br>"
            f"开盘：{r['open']:.4f}<br>"
            f"最高：{r['high']:.4f}<br>"
            f"最低：{r['low']:.4f}<br>"
            f"收盘：{r['close']:.4f}"
        )
        for _, r in df.iterrows()
    ]

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing=dict(
                line=dict(color=UP_COLOR, width=1.2),
                fillcolor="rgba(255, 255, 255, 0)"
            ),
            decreasing=dict(
                line=dict(color=DOWN_COLOR, width=1.2),
                fillcolor=DOWN_COLOR
            ),
            hovertext=k_hover_text,
            hoverinfo="text"
        ),
        row=1,
        col=1
    )

    # MA均线
    for window in MA_WINDOWS:
        ma_col = f"MA{window}"
        if ma_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[ma_col],
                    mode="lines",
                    name=ma_col,
                    line=dict(
                        width=0.8,
                        color=MA_COLORS.get(window, None)
                    ),
                    hovertemplate=f"{ma_col}：%{{y:.4f}}<extra></extra>"
                ),
                row=1,
                col=1
            )

    # 回测买入卖出 B/S 标记：annotation 放在K线上下方，箭头指向交易价格
    if trade_df is not None and not trade_df.empty:
        mark_df = trade_df.copy()
        mark_df["日期"] = pd.to_datetime(mark_df["日期"])

        buy_df = mark_df[mark_df["操作"] == "买入"].copy()
        sell_df = mark_df[mark_df["操作"].isin(["卖出", "强制清仓"])].copy()

        # 买入 B：放在价格点下方，箭头指向当日开盘价
        for _, r in buy_df.iterrows():
            fig.add_annotation(
                x=r["日期"],
                y=r["价格"],
                text="B",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=UP_COLOR,
                ax=0,
                ay=42,
                bgcolor=UP_COLOR,
                bordercolor=UP_COLOR,
                borderwidth=1,
                font=dict(color="white", size=10),
                xanchor="center",
                yanchor="middle",
                row=1,
                col=1
            )

        # 卖出 S：放在价格点上方，箭头指向当日收盘价
        for _, r in sell_df.iterrows():
            fig.add_annotation(
                x=r["日期"],
                y=r["价格"],
                text="S",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=DOWN_COLOR,
                ax=0,
                ay=-42,
                bgcolor=DOWN_COLOR,
                bordercolor=DOWN_COLOR,
                borderwidth=1,
                font=dict(color="white", size=10),
                xanchor="center",
                yanchor="middle",
                row=1,
                col=1
            )

        # 透明点用于显示回测交易 hover 明细
        if not buy_df.empty:
            buy_hover_text = [
                (
                    f"日期：{pd.to_datetime(r['日期']).strftime('%Y/%m/%d')}<br>"
                    f"操作：买入 B<br>"
                    f"买入价：{r['价格']:.4f}<br>"
                    f"原因：{r['原因']}<br>"
                    f"成交份额：{r['成交份额']:,.2f}<br>"
                    f"现金：{r['现金']:,.2f}<br>"
                    f"持仓份额：{r['持仓份额']:,.2f}<br>"
                    f"总资产：{r['总资产']:,.2f}"
                )
                for _, r in buy_df.iterrows()
            ]

            fig.add_trace(
                go.Scatter(
                    x=buy_df["日期"],
                    y=buy_df["价格"],
                    mode="markers",
                    marker=dict(size=14, color="rgba(0,0,0,0)"),
                    name="买入明细",
                    hovertext=buy_hover_text,
                    hovertemplate="%{hovertext}<extra></extra>",
                    showlegend=False
                ),
                row=1,
                col=1
            )

        if not sell_df.empty:
            sell_hover_text = [
                (
                    f"日期：{pd.to_datetime(r['日期']).strftime('%Y/%m/%d')}<br>"
                    f"操作：{r['操作']} S<br>"
                    f"卖出价：{r['价格']:.4f}<br>"
                    f"原因：{r['原因']}<br>"
                    f"成交份额：{r['成交份额']:,.2f}<br>"
                    f"现金：{r['现金']:,.2f}<br>"
                    f"持仓份额：{r['持仓份额']:,.2f}<br>"
                    f"总资产：{r['总资产']:,.2f}"
                )
                for _, r in sell_df.iterrows()
            ]

            fig.add_trace(
                go.Scatter(
                    x=sell_df["日期"],
                    y=sell_df["价格"],
                    mode="markers",
                    marker=dict(size=14, color="rgba(0,0,0,0)"),
                    name="卖出明细",
                    hovertext=sell_hover_text,
                    hovertemplate="%{hovertext}<extra></extra>",
                    showlegend=False
                ),
                row=1,
                col=1
            )

    # 成交量
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            marker_color=volume_colors,
            name="成交量",
            hovertemplate=(
                "日期：%{x|%Y/%m/%d}<br>"
                "成交量：%{y:,.0f}"
                "<extra></extra>"
            )
        ),
        row=2,
        col=1
    )

    # MACD
    macd_golden_df = df[df["MACD_GOLDEN"]]
    macd_death_df = df[df["MACD_DEATH"]]

    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["MACD"],
            marker_color=macd_colors,
            name="MACD柱",
            hovertemplate=(
                "日期：%{x|%Y/%m/%d}<br>"
                "MACD柱：%{y:.6f}"
                "<extra></extra>"
            )
        ),
        row=3,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["DIF"],
            mode="lines",
            name="DIF",
            line=dict(width=1),
            hovertemplate="DIF：%{y:.6f}<extra></extra>"
        ),
        row=3,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["DEA"],
            mode="lines",
            name="DEA",
            line=dict(width=1),
            hovertemplate="DEA：%{y:.6f}<extra></extra>"
        ),
        row=3,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=macd_golden_df["date"],
            y=macd_golden_df["DIF"],
            mode="markers",
            marker=dict(
                symbol="circle-open",
                size=6,
                color=UP_COLOR,
                line=dict(width=0.8, color=UP_COLOR)
            ),
            name="MACD金叉",
            hovertemplate=(
                "日期：%{x|%Y/%m/%d}<br>"
                "MACD金叉<br>"
                "DIF：%{y:.6f}"
                "<extra></extra>"
            )
        ),
        row=3,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=macd_death_df["date"],
            y=macd_death_df["DIF"],
            mode="markers",
            marker=dict(
                symbol="x-thin",
                size=7,
                color=DOWN_COLOR,
                line=dict(width=0.8, color=DOWN_COLOR)
            ),
            name="MACD死叉",
            hovertemplate=(
                "日期：%{x|%Y/%m/%d}<br>"
                "MACD死叉<br>"
                "DIF：%{y:.6f}"
                "<extra></extra>"
            )
        ),
        row=3,
        col=1
    )

    # KDJ
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["K"],
            mode="lines",
            name="K",
            line=dict(width=1),
            hovertemplate="K：%{y:.4f}<extra></extra>"
        ),
        row=4,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["D"],
            mode="lines",
            name="D",
            line=dict(width=1),
            hovertemplate="D：%{y:.4f}<extra></extra>"
        ),
        row=4,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["J"],
            mode="lines",
            name="J",
            line=dict(width=1),
            hovertemplate="J：%{y:.4f}<extra></extra>"
        ),
        row=4,
        col=1
    )

    fig.add_hline(
        y=overbought,
        line_dash="dash",
        line_color="red",
        line_width=0.8,
        annotation_text="超买区",
        row=4,
        col=1
    )

    fig.add_hline(
        y=oversold,
        line_dash="dash",
        line_color="green",
        line_width=0.8,
        annotation_text="超卖区",
        row=4,
        col=1
    )

    kdj_golden_df = df[df["KDJ_GOLDEN"]]
    kdj_death_df = df[df["KDJ_DEATH"]]

    fig.add_trace(
        go.Scatter(
            x=kdj_golden_df["date"],
            y=kdj_golden_df["K"],
            mode="markers",
            marker=dict(
                symbol="triangle-up-open",
                size=7,
                color=UP_COLOR,
                line=dict(width=0.8, color=UP_COLOR)
            ),
            name="KDJ金叉",
            hovertemplate=(
                "日期：%{x|%Y/%m/%d}<br>"
                "KDJ金叉<br>"
                "K：%{y:.4f}"
                "<extra></extra>"
            )
        ),
        row=4,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=kdj_death_df["date"],
            y=kdj_death_df["K"],
            mode="markers",
            marker=dict(
                symbol="triangle-down-open",
                size=7,
                color=DOWN_COLOR,
                line=dict(width=0.8, color=DOWN_COLOR)
            ),
            name="KDJ死叉",
            hovertemplate=(
                "日期：%{x|%Y/%m/%d}<br>"
                "KDJ死叉<br>"
                "K：%{y:.4f}"
                "<extra></extra>"
            )
        ),
        row=4,
        col=1
    )

    fig.update_layout(
        height=980,
        hovermode="x unified",
        spikedistance=-1,
        hoverdistance=100,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="KDJ", row=4, col=1)

    # 所有子图同步显示竖向虚线
    fig.update_xaxes(
        tickformat="%Y/%m/%d",
        hoverformat="%Y/%m/%d",
        showspikes=True,
        spikecolor="rgba(80,80,80,0.65)",
        spikethickness=1,
        spikedash="dot",
        spikemode="across",
        spikesnap="cursor"
    )

    fig.update_xaxes(title_text="日期", row=4, col=1)

    return fig


# =========================
# 回测逻辑
# =========================

def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    start_date,
    end_date,
    initial_cash: float,
    buy_pct: float,
    sell_pct: float,
    overbought: float,
    oversold: float
):
    bt = df[
        (df["date"].dt.date >= start_date) &
        (df["date"].dt.date <= end_date)
    ].copy()

    if bt.empty:
        return None, None, pd.DataFrame(), pd.DataFrame()

    cash = initial_cash
    shares = 0.0

    trade_records = []
    asset_records = []

    # 买入持有基准：开始日收盘买入，结束日收盘卖出
    first_row = bt.iloc[0]
    last_row = bt.iloc[-1]

    benchmark_buy_price = first_row["close"]
    benchmark_sell_price = last_row["close"]

    benchmark_shares = initial_cash / benchmark_buy_price if benchmark_buy_price > 0 else 0
    benchmark_final_asset = benchmark_shares * benchmark_sell_price
    benchmark_profit_amount = benchmark_final_asset - initial_cash
    benchmark_profit_ratio = benchmark_profit_amount / initial_cash if initial_cash != 0 else 0

    benchmark_records = []

    for _, row in bt.iterrows():
        benchmark_asset = benchmark_shares * row["close"]
        benchmark_records.append({
            "日期": row["date"].date(),
            "基准收盘价": row["close"],
            "基准持仓份额": benchmark_shares,
            "基准总资产": benchmark_asset
        })

    benchmark_df = pd.DataFrame(benchmark_records)

    benchmark_summary = {
        "基准策略": "开始日期收盘价全仓买入，结束日期收盘价全仓卖出",
        "基准买入日期": first_row["date"].date(),
        "基准卖出日期": last_row["date"].date(),
        "基准买入价格": benchmark_buy_price,
        "基准卖出价格": benchmark_sell_price,
        "基准持仓份额": benchmark_shares,
        "基准最终资产": benchmark_final_asset,
        "基准盈亏金额": benchmark_profit_amount,
        "基准盈亏比例": benchmark_profit_ratio
    }

    # 策略回测
    bt = bt.reset_index(drop=True)

    for idx, row in bt.iterrows():
        current_date = row["date"]
        buy_price = row["open"]      # 策略买入使用当日开盘价
        sell_price = row["close"]    # 策略卖出使用当日收盘价
        close_price = row["close"]

        if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0 or sell_price <= 0:
            continue

        buy_signal = False
        sell_signal = False
        reason = ""

        if strategy_name == "MACD金叉死叉策略":
            buy_signal = bool(row["MACD_GOLDEN"])
            sell_signal = bool(row["MACD_DEATH"])

            if buy_signal:
                reason = "MACD金叉买入"
            elif sell_signal:
                reason = "MACD死叉卖出"

        elif strategy_name == "KDJ策略":
            # 正确逻辑：超卖区金叉买入，超买区死叉卖出
            is_oversold = row["K"] <= oversold and row["D"] <= oversold
            is_overbought = row["K"] >= overbought and row["D"] >= overbought

            buy_signal = bool(row["KDJ_GOLDEN"] and is_oversold)
            sell_signal = bool(row["KDJ_DEATH"] and is_overbought)

            if buy_signal:
                reason = "KDJ超卖区金叉买入"
            elif sell_signal:
                reason = "KDJ超买区死叉卖出"

        # 买入：使用当日开盘价
        if buy_signal and cash > 0:
            buy_amount = cash * buy_pct
            buy_shares = buy_amount / buy_price

            cash -= buy_amount
            shares += buy_shares

            trade_records.append({
                "日期": current_date.date(),
                "操作": "买入",
                "原因": reason,
                "价格": buy_price,
                "成交份额": buy_shares,
                "现金": cash,
                "持仓份额": shares,
                "总资产": cash + shares * close_price
            })

        # 卖出：使用当日收盘价
        if sell_signal and shares > 0:
            sell_shares = shares * sell_pct
            sell_amount = sell_shares * sell_price

            shares -= sell_shares
            cash += sell_amount

            trade_records.append({
                "日期": current_date.date(),
                "操作": "卖出",
                "原因": reason,
                "价格": sell_price,
                "成交份额": sell_shares,
                "现金": cash,
                "持仓份额": shares,
                "总资产": cash + shares * close_price
            })

        # 到结束日期强制清仓：使用结束日收盘价
        is_last_day = idx == len(bt) - 1

        if is_last_day and shares > 0:
            final_sell_amount = shares * sell_price

            trade_records.append({
                "日期": current_date.date(),
                "操作": "强制清仓",
                "原因": "到达终止日期，按收盘价卖出全部份额",
                "价格": sell_price,
                "成交份额": shares,
                "现金": cash + final_sell_amount,
                "持仓份额": 0,
                "总资产": cash + final_sell_amount
            })

            cash += final_sell_amount
            shares = 0

        total_asset = cash + shares * close_price

        asset_records.append({
            "日期": current_date.date(),
            "收盘价": close_price,
            "现金": cash,
            "持仓份额": shares,
            "持仓市值": shares * close_price,
            "策略总资产": total_asset
        })

    final_asset = cash
    profit_amount = final_asset - initial_cash
    profit_ratio = profit_amount / initial_cash if initial_cash != 0 else 0

    strategy_summary = {
        "初始金额": initial_cash,
        "最终资产": final_asset,
        "盈亏金额": profit_amount,
        "盈亏比例": profit_ratio,
        "交易次数": len(trade_records),
        "开始日期": start_date,
        "结束日期": end_date,
        "策略": strategy_name
    }

    trade_df = pd.DataFrame(trade_records)
    asset_df = pd.DataFrame(asset_records)

    if not asset_df.empty and not benchmark_df.empty:
        asset_df = asset_df.merge(
            benchmark_df[["日期", "基准总资产"]],
            on="日期",
            how="left"
        )

    return strategy_summary, benchmark_summary, trade_df, asset_df


def plot_asset_curve(asset_df: pd.DataFrame):
    fig = go.Figure()

    if "策略总资产" in asset_df.columns:
        fig.add_trace(
            go.Scatter(
                x=asset_df["日期"],
                y=asset_df["策略总资产"],
                mode="lines",
                name="策略总资产",
                line=dict(width=1.3, color="#1f77b4"),
                hovertemplate=(
                    "日期：%{x|%Y/%m/%d}<br>"
                    "策略总资产：%{y:,.2f}"
                    "<extra></extra>"
                )
            )
        )

    if "基准总资产" in asset_df.columns:
        fig.add_trace(
            go.Scatter(
                x=asset_df["日期"],
                y=asset_df["基准总资产"],
                mode="lines",
                name="买入持有基准",
                line=dict(
                    width=1.0,
                    color=UP_COLOR
                ),
                hovertemplate=(
                    "日期：%{x|%Y/%m/%d}<br>"
                    "买入持有基准：%{y:,.2f}"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        height=350,
        title="回测资产曲线：策略 vs 买入持有基准",
        xaxis_title="日期",
        yaxis_title="总资产",
        hovermode="x unified",
        spikedistance=-1,
        hoverdistance=100
    )

    fig.update_xaxes(
        tickformat="%Y/%m/%d",
        hoverformat="%Y/%m/%d",
        showspikes=True,
        spikecolor="rgba(80,80,80,0.65)",
        spikethickness=1,
        spikedash="dot",
        spikemode="across",
        spikesnap="cursor"
    )

    return fig


# =========================
# Streamlit 页面主体
# =========================

st.title("银行ETF天弘 SH515290 量化回测看板")

st.caption(
    "说明：本工具仅用于技术学习和策略回测验证，不构成任何投资建议。"
)


# =========================
# 侧边栏：数据设置
# =========================

st.sidebar.header("数据设置")

symbol = st.sidebar.text_input(
    "投资标的",
    value="SH515290"
)

data_start = st.sidebar.date_input(
    "行情开始日期",
    value=date(2020, 1, 1)
)

data_end = st.sidebar.date_input(
    "行情结束日期",
    value=date.today()
)

adjust_options = {
    "前复权": "qfq",
    "后复权": "hfq",
    "不复权": ""
}

adjust_name = st.sidebar.selectbox(
    "复权方式",
    options=list(adjust_options.keys()),
    index=0
)

adjust = adjust_options[adjust_name]

data_source = st.sidebar.radio(
    "数据来源",
    options=[
        "优先在线获取，失败后使用本地缓存",
        "仅使用本地缓存",
        "上传CSV"
    ],
    index=0
)

uploaded_file = None

if data_source == "上传CSV":
    uploaded_file = st.sidebar.file_uploader(
        "上传行情CSV",
        type=["csv"]
    )

st.sidebar.header("KDJ参数")

overbought = st.sidebar.slider(
    "超买线",
    min_value=50,
    max_value=100,
    value=80,
    step=1
)

oversold = st.sidebar.slider(
    "超卖线",
    min_value=0,
    max_value=50,
    value=20,
    step=1
)

start_str = data_start.strftime("%Y%m%d")
end_str = data_end.strftime("%Y%m%d")


# =========================
# 数据加载
# =========================

raw_df = pd.DataFrame()

if data_source == "上传CSV":
    if uploaded_file is None:
        st.warning("请在左侧上传 CSV 文件。")
        st.stop()

    try:
        raw_df = load_uploaded_csv(uploaded_file)
        save_local_cache(raw_df)
        st.success("已成功读取上传的 CSV，并已更新本地缓存。")
    except Exception as e:
        st.error(f"CSV读取失败：{e}")
        st.stop()

elif data_source == "仅使用本地缓存":
    raw_df = read_local_cache()

    if raw_df.empty:
        st.error(
            f"未找到可用本地缓存文件：{CACHE_FILE}。请先在线获取一次，或上传 CSV。"
        )
        st.stop()

    st.info("当前使用本地缓存数据。")

else:
    try:
        raw_df = load_etf_daily_data_online(
            symbol=symbol,
            start_date=start_str,
            end_date=end_str,
            adjust=adjust
        )

        save_local_cache(raw_df)
        st.success("在线行情数据获取成功，并已保存为本地缓存。")

    except Exception as e:
        st.warning(f"在线数据获取失败：{e}")

        cache_df = read_local_cache()

        if cache_df.empty:
            st.error(
                "在线获取失败，且没有可用本地缓存。请稍后重试，或选择“上传CSV”。"
            )
            st.stop()
        else:
            raw_df = cache_df
            st.info("已自动使用本地缓存数据继续运行。")


if raw_df.empty:
    st.warning("未获取到行情数据，请检查数据来源、日期范围或 CSV 格式。")
    st.stop()


raw_df = raw_df[
    (raw_df["date"].dt.date >= data_start) &
    (raw_df["date"].dt.date <= data_end)
].copy()

if raw_df.empty:
    st.warning("当前日期范围内没有行情数据。")
    st.stop()


df = calculate_indicators(raw_df)
latest = df.iloc[-1]


# =========================
# 顶部指标
# =========================

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("最新日期", latest["date"].strftime("%Y/%m/%d"))
col2.metric("最新收盘价", f"{latest['close']:.4f}")

if "pct_change" in latest.index and pd.notna(latest["pct_change"]):
    col3.metric("涨跌幅", f"{latest['pct_change']:.2f}%")
else:
    col3.metric("涨跌幅", "N/A")

col4.metric("成交量", f"{latest['volume']:,.0f}")
col5.metric("MACD", f"{latest['MACD']:.4f}")


# =========================
# 回测设置
# =========================

st.subheader("策略回测")

min_trade_date = df["date"].dt.date.min()
max_trade_date = df["date"].dt.date.max()

bt_col1, bt_col2, bt_col3, bt_col4, bt_col5 = st.columns(5)

with bt_col1:
    backtest_start = st.date_input(
        "回测开始日期",
        value=min_trade_date,
        min_value=min_trade_date,
        max_value=max_trade_date
    )

with bt_col2:
    backtest_end = st.date_input(
        "回测结束日期",
        value=max_trade_date,
        min_value=min_trade_date,
        max_value=max_trade_date
    )

with bt_col3:
    initial_cash = st.number_input(
        "初始金额",
        min_value=1000.0,
        value=100000.0,
        step=1000.0
    )

with bt_col4:
    buy_pct_input = st.number_input(
        "买入百分比 %",
        min_value=1.0,
        max_value=100.0,
        value=100.0,
        step=1.0
    )

with bt_col5:
    sell_pct_input = st.number_input(
        "卖出百分比 %",
        min_value=1.0,
        max_value=100.0,
        value=100.0,
        step=1.0
    )

strategy_name = st.selectbox(
    "选择策略",
    options=[
        "MACD金叉死叉策略",
        "KDJ策略"
    ]
)

buy_pct = buy_pct_input / 100
sell_pct = sell_pct_input / 100

if backtest_start > backtest_end:
    st.error("回测开始日期不能晚于结束日期。")
    st.stop()

summary, benchmark_summary, trade_df, asset_df = run_backtest(
    df=df,
    strategy_name=strategy_name,
    start_date=backtest_start,
    end_date=backtest_end,
    initial_cash=initial_cash,
    buy_pct=buy_pct,
    sell_pct=sell_pct,
    overbought=overbought,
    oversold=oversold
)

if summary is None:
    st.warning("当前回测区间没有可用数据。")
    st.stop()


# =========================
# 行情与指标图
# =========================

st.subheader("行情与技术指标")

fig = plot_dashboard(
    df=df,
    trade_df=trade_df,
    overbought=overbought,
    oversold=oversold
)

st.plotly_chart(fig, use_container_width=True)


# =========================
# 回测结果展示
# =========================

st.markdown("### 策略结果")

result_col1, result_col2, result_col3, result_col4 = st.columns(4)

result_col1.metric("策略初始金额", f"{summary['初始金额']:,.2f}")
result_col2.metric("策略最终资产", f"{summary['最终资产']:,.2f}")
result_col3.metric("策略盈亏金额", f"{summary['盈亏金额']:,.2f}")
result_col4.metric("策略盈亏比例", f"{summary['盈亏比例'] * 100:.2f}%")

st.write(f"当前策略：**{summary['策略']}**")

strategy_description = get_strategy_description(
    strategy_name=strategy_name,
    buy_pct_input=buy_pct_input,
    sell_pct_input=sell_pct_input,
    overbought=overbought,
    oversold=oversold
)

st.info(strategy_description)

st.write(f"交易次数：**{summary['交易次数']}**")


st.markdown("### 买入持有基准")

bench_col1, bench_col2, bench_col3, bench_col4 = st.columns(4)

bench_col1.metric(
    "基准最终资产",
    f"{benchmark_summary['基准最终资产']:,.2f}"
)

bench_col2.metric(
    "基准盈亏金额",
    f"{benchmark_summary['基准盈亏金额']:,.2f}"
)

bench_col3.metric(
    "基准盈亏比例",
    f"{benchmark_summary['基准盈亏比例'] * 100:.2f}%"
)

bench_col4.metric(
    "基准持仓份额",
    f"{benchmark_summary['基准持仓份额']:,.2f}"
)

st.write(
    f"基准逻辑：**{benchmark_summary['基准策略']}**；"
    f"买入日期：**{benchmark_summary['基准买入日期'].strftime('%Y/%m/%d')}**，"
    f"买入价格：**{benchmark_summary['基准买入价格']:.4f}**；"
    f"卖出日期：**{benchmark_summary['基准卖出日期'].strftime('%Y/%m/%d')}**，"
    f"卖出价格：**{benchmark_summary['基准卖出价格']:.4f}**。"
)


st.markdown("### 策略 vs 买入持有对比")

diff_final_asset = summary["最终资产"] - benchmark_summary["基准最终资产"]
diff_profit_amount = summary["盈亏金额"] - benchmark_summary["基准盈亏金额"]
diff_profit_ratio = summary["盈亏比例"] - benchmark_summary["基准盈亏比例"]

compare_col1, compare_col2, compare_col3 = st.columns(3)

compare_col1.metric(
    "最终资产差额",
    f"{diff_final_asset:,.2f}",
    delta=f"{diff_final_asset:,.2f}"
)

compare_col2.metric(
    "盈亏金额差额",
    f"{diff_profit_amount:,.2f}",
    delta=f"{diff_profit_amount:,.2f}"
)

compare_col3.metric(
    "收益率差额",
    f"{diff_profit_ratio * 100:.2f}%",
    delta=f"{diff_profit_ratio * 100:.2f}%"
)

comparison_df = pd.DataFrame([
    {
        "类型": "策略回测",
        "最终资产": summary["最终资产"],
        "盈亏金额": summary["盈亏金额"],
        "盈亏比例": summary["盈亏比例"],
        "交易次数": summary["交易次数"]
    },
    {
        "类型": "买入持有基准",
        "最终资产": benchmark_summary["基准最终资产"],
        "盈亏金额": benchmark_summary["基准盈亏金额"],
        "盈亏比例": benchmark_summary["基准盈亏比例"],
        "交易次数": 2
    }
])

comparison_df["盈亏比例"] = comparison_df["盈亏比例"].apply(lambda x: f"{x * 100:.2f}%")

st.dataframe(comparison_df, use_container_width=True)


# =========================
# 资产曲线
# =========================

if not asset_df.empty:
    st.subheader("回测资产曲线")
    asset_fig = plot_asset_curve(asset_df)
    st.plotly_chart(asset_fig, use_container_width=True)


# =========================
# 交易明细
# =========================

if trade_df.empty:
    st.info("当前回测区间内没有触发交易信号。")
else:
    st.subheader("交易明细")
    st.dataframe(trade_df, use_container_width=True)

    csv_data = trade_df.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="下载交易明细 CSV",
        data=csv_data,
        file_name="backtest_trades.csv",
        mime="text/csv"
    )


# =========================
# 数据查看与下载
# =========================

with st.expander("查看行情与指标数据"):
    show_cols = [
        "date", "open", "high", "low", "close", "volume",
        "amount", "pct_change",
        "MA5", "MA10", "MA20", "MA30", "MA60", "MA120", "MA250", "MA300",
        "DIF", "DEA", "MACD",
        "K", "D", "J",
        "MACD_GOLDEN", "MACD_DEATH",
        "KDJ_GOLDEN", "KDJ_DEATH"
    ]

    available_cols = [col for col in show_cols if col in df.columns]

    display_df = df[available_cols].copy()

    if "date" in display_df.columns:
        display_df["date"] = display_df["date"].dt.strftime("%Y/%m/%d")

    st.dataframe(
        display_df,
        use_container_width=True
    )

    full_csv = display_df.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="下载行情与指标数据 CSV",
        data=full_csv,
        file_name="market_indicator_data.csv",
        mime="text/csv"
    )