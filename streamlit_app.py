import functools
import json
import logging
import math
import os
import pickle
from datetime import date
from pathlib import Path

import cufflinks as cf
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import streamlit as st
import yfinance as yf
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode

logger = logging.getLogger(__name__)

setattr(plotly.offline, "__PLOTLY_OFFLINE_INITIALIZED", True)

logger.info("appp start")

# Global variables
const_date_range = 120
if "input_init_funds" not in st.session_state:
    st.session_state["input_init_funds"] = 100000

if "account_summary" not in st.session_state:
    st.session_state["account_summary"] = {
        "total": st.session_state["input_init_funds"],
        "total_delta": 0,
        "total_percent": 100,
        "trade_sequence": 0,
        "available_funds": st.session_state["input_init_funds"],
        "current_holding_shares": 0,
        "current_holding_value": 0,
        "current_avg_trade_cost": 0,
    }

if "log_text" not in st.session_state:
    st.session_state["log_text"] = ""

if "trade_price" not in st.session_state:
    st.session_state["trade_price"] = "next day"

if "stop_price" not in st.session_state:
    st.session_state["stop_price"] = "10%"

if "filter_start" not in st.session_state:
    st.session_state["filter_start"] = 0
    st.session_state["filter_end"] = const_date_range


@st.cache_data
def load_data(ticker: str, start: str) -> pd.DataFrame:
    # todo change to load data from yfinance
    msft = yf.Ticker(ticker)
    # get all stock info (slow)
    # info = msft.info
    # json.dump(info.info, open(f"./data/{ticker}_info.json", "w"), indent=4)

    # if
    cache = f"./data/{ticker}.parquet"
    if os.path.exists(cache):
        hist1 = pd.read_parquet(cache)
        t1 = hist1.index[-1]
        hist2 = hist = msft.history(period="1d", start=t1)
        hist = pd.concat([hist1[:-1], hist2])
    else:
        hist = msft.history(period="1d", start=start)

    df = pd.DataFrame(hist)
    df.to_parquet(f"./data/{ticker}.parquet")

    # df = cf.datagen.ohlcv()
    return df


def get_chart(df: pd.DataFrame):
    qf = cf.QuantFig(df, title="Storck Quant Figure", legend="top", name="GS")

    qf.add_sma(
        [5, 10, 55], width=2, colors=["green", "orange", "blue"], legendgroup=True
    )
    # qf.add_rsi(periods=20, color="java")
    # qf.add_bollinger_bands(
    #     periods=20, boll_std=2, colors=["magenta", "grey"], fill=True
    # )
    qf.add_volume()
    qf.add_macd()
    # qf.add_trendline()
    fig = qf.iplot(asFigure=True)
    # st.plotly_chart(fig)
    return fig


@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take Raw Fidelity Dataframe and return usable dataframe.
    - snake_case headers
    - Include 401k by filling na type
    - Drop Cash accounts and misc text
    - Clean $ and % signs from values and convert to floats

    Args:
        df (pd.DataFrame): Raw fidelity csv data

    Returns:
        pd.DataFrame: cleaned dataframe with features above
    """
    # df = df.copy()
    # df.columns = df.columns.str.lower().str.replace(" ", "_", regex=False).str.replace("/", "_", regex=False)

    # df.type = df.type.fillna("unknown")
    # df = df.dropna()

    # price_index = df.columns.get_loc("last_price")
    # cost_basis_index = df.columns.get_loc("cost_basis_per_share")
    # df[df.columns[price_index : cost_basis_index + 1]] = df[
    #     df.columns[price_index : cost_basis_index + 1]
    # ].transform(lambda s: s.str.replace("$", "", regex=False).str.replace("%", "", regex=False).astype(float))

    # quantity_index = df.columns.get_loc("quantity")
    # most_relevant_columns = df.columns[quantity_index : cost_basis_index + 1]
    # first_columns = df.columns[0:quantity_index]
    # last_columns = df.columns[cost_basis_index + 1 :]
    # df = df[[*most_relevant_columns, *first_columns, *last_columns]]
    return df


def filter_data(df: pd.DataFrame, start: int = None, end: int = None) -> pd.DataFrame:
    """
    Returns Dataframe with only accounts and symbols selected

    Args:
        df (pd.DataFrame): clean fidelity csv data, including account_name and symbol columns
        account_selections (list[str]): list of account names to include
        symbol_selections (list[str]): list of symbols to include

    Returns:
        pd.DataFrame: data only for the given accounts and symbols
    """
    global const_date_range
    # filter data
    # item =  df.index[0]
    # i_start = pd.Timestamp(start,tz=item.tz)
    # df = df[df.index >= i_start]

    if start is None:
        start = 0

    if end is None:
        end = const_date_range

    df = df.iloc[start:end]
    return df


def update_init_fund():
    st.session_state["account_summary"] = {
        "total": st.session_state["input_init_funds"],
        "total_delta": 0,
        "total_percent": 100,
    }


def construct_sidebar():
    st.sidebar.subheader("Account Summary")
    st.sidebar.metric(
        "Total",
        f"${st.session_state.account_summary['total']:,.2f}",
        f"{st.session_state.account_summary['total_delta']:.2f}",
    )

    st.sidebar.metric(
        "Percentage",
        f"${st.session_state.account_summary['total_percent']}%",
    )

    st.sidebar.subheader("Settings")
    option = st.sidebar.selectbox(
        "Ticker",
        ("MSFT", "SQQQ", "TQQQ"),
        key="input_ticker",
    )

    d = st.sidebar.date_input("Start Date", date(2010, 1, 1), key="input_date_start")
    # st.sidebar.write('Your birthday is:', d)

    st.sidebar.number_input(
        "Initial Funds",
        min_value=10000,
        max_value=1000000,
        value=st.session_state.input_init_funds,
        format="%d",
        step=10000,
        key="input_init_funds",
        on_change=update_init_fund,
    )

    st.sidebar.number_input(
        "Funds/trade", value=10000, format="%d", key="input_fund_per_trade", step=1000
    )

    st.sidebar.radio("Trade Price", ("same day", "next day"), key="trade_price")
    st.sidebar.radio("Stop Price", ("20%", "15%", "10%", "8%", "5%"), key="stop_price")
    st.sidebar.number_input(
        "Trade Fee",
        min_value=0,
        max_value=100,
        value=st.session_state.get("trade_fee", 10),
        key="trade_fee",
    )


def construct_chart_section(df: pd.DataFrame):
    item = df.iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)
    account_summary = st.session_state.account_summary
    with col1:
        st.write("Ticker:", st.session_state.input_ticker)
    with col2:
        st.write("Current Date:", st.session_state.input_date_end)
    with col3:
        st.write("Current Price:", round(item.get("Close"), 4))
    with col4:
        st.write("Holding Shares:", account_summary["current_holding_shares"])
    with col5:
        st.write("Holding Value:", round(account_summary["current_holding_value"], 2))

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.button("Next Day", key="btn_next_day", on_click=btn_next_day_click)
    with col2:
        st.button("Next 5 Days", key="btn_next_5days", on_click=btn_next_5days_click)
    with col3:
        st.write("Stop Price:", round(account_summary.get("stop_price", 0), 4))
    with col4:
        st.write(
            "Avg Cost:", round(account_summary.get("current_avg_trade_cost", 0), 4)
        )
    with col5:
        st.write("Trade Message:", st.session_state.get("trade_message", ""))

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.button("Buy", key="btn_buy", on_click=btn_buy_click)
    with col2:
        st.button("Sell", key="btn_sell", on_click=btn_sell_click)
    with col3:
        st.write("Low Price:", round(item.get("Low"), 4))

    fig = get_chart(df)
    fig.update_layout(height=1000)
    st.plotly_chart(fig, use_container_width=True)


def p2f(x):
    return float(x.strip("%")) / 100


def check_stop_price():
    global df, df2

    account_summary = st.session_state["account_summary"]
    if account_summary["current_holding_shares"] == 0:
        # nothing to do
        return

    i_end = st.session_state.filter_end
    current_item = df2.iloc[-1]
    next_item = df.iloc[i_end + 1]
    price = current_item["Low"]
    trade_fee = st.session_state.trade_fee
    stop_price = account_summary["stop_price"]

    if price < stop_price:
        # sell
        cash_sold = account_summary["current_holding_shares"] * price
        account_summary["available_funds"] += cash_sold - trade_fee
        account_summary["current_holding_shares"] = 0
        account_summary["current_avg_trade_cost"] = 0
        account_summary["stop_price"] = 0

        st.session_state["account_summary"] = account_summary
        st.session_state["trade_message"] = "Sucessful"
        cal_account_summary()


def btn_sell_click():
    global df, df2
    i_end = st.session_state.filter_end
    current_item = df2.iloc[-1]
    next_item = df.iloc[i_end + 1]
    if st.session_state.trade_price == "same day":
        price = current_item["Low"]
    else:
        price = next_item["Open"]
    trade_fee = st.session_state.trade_fee

    account_summary = st.session_state["account_summary"]
    cash_sold = account_summary["current_holding_shares"] * price
    account_summary["available_funds"] += cash_sold - trade_fee
    account_summary["current_holding_shares"] = 0
    account_summary["current_avg_trade_cost"] = 0
    account_summary["stop_price"] = 0

    st.session_state["account_summary"] = account_summary
    st.session_state["trade_message"] = "Sucessful"
    cal_account_summary()


def btn_buy_click():
    global df, df2
    i_end = st.session_state.filter_end
    current_item = df2.iloc[-1]
    next_item = df.iloc[i_end + 1]
    if st.session_state.trade_price == "same day":
        price = current_item["High"]
    else:
        price = next_item["Open"]

    account_summary = st.session_state["account_summary"]
    available_funds = account_summary["available_funds"]
    trading_funds = st.session_state["input_fund_per_trade"]
    if available_funds < trading_funds:
        logger.warning("not enough funds")
        st.session_state["trade_message"] = "Not enough Funds"
        return

    account_summary["trade_sequence"] += 1
    trade_fee = st.session_state.trade_fee
    new_shares = math.floor(trading_funds / price)
    existing_shares = account_summary["current_holding_shares"]
    existing_cost = account_summary["current_avg_trade_cost"]
    new_cost = (existing_shares * existing_cost + new_shares * price + trade_fee) / (
        existing_shares + new_shares
    )
    account_summary["current_avg_trade_cost"] = new_cost
    account_summary["current_holding_shares"] = existing_shares + new_shares
    account_summary["available_funds"] = (
        account_summary["available_funds"]
        - new_shares * price
        - st.session_state.trade_fee
    )
    stop_price_percent = p2f(st.session_state.get("stop_price", "10%"))
    account_summary["stop_price"] = price * (1 - stop_price_percent)

    st.session_state["account_summary"] = account_summary
    st.session_state["trade_message"] = "Sucessful"
    cal_account_summary()

    # {
    #     "total": st.session_state["input_init_funds"],
    #     "total_delta": 0,
    #     "total_percent": 100,
    #     "trade_sequence": 0,
    #     "current_holding_shares":0,
    #     "current_holding_value":0,
    #     "current_avg_trade_cost":0
    # }


def cal_account_summary():
    global df, df2
    i_end = st.session_state.filter_end
    current_item = df2.iloc[-1]
    next_item = df.iloc[i_end + 1]
    price = current_item["Close"]
    account_summary = st.session_state.account_summary
    account_summary["current_price"] = price
    account_summary["current_holding_value"] = (
        account_summary["current_holding_shares"] * price
    )
    account_summary["total"] = (
        account_summary["available_funds"]
        + account_summary["current_holding_shares"] * price
    )

    account_summary["total_delta"] = (
        account_summary["total"] - st.session_state["input_init_funds"]
    )
    account_summary["total_percent"] = round(
        account_summary["total"] * 100 / st.session_state["input_init_funds"], 2
    )
    st.session_state.account_summary = account_summary


def btn_next_day_click():
    global const_date_range
    i_start = st.session_state.get("filter_start", 0)
    i_end = st.session_state.get("filter_end", const_date_range)
    if i_end - i_start < const_date_range:
        st.session_state.filter_end = i_end + 1
    else:
        st.session_state.filter_start = i_start + 1
        st.session_state.filter_end = i_end + 1

    cal_account_summary()


def btn_next_5days_click():
    global const_date_range
    i_start = st.session_state.get("filter_start", 0)
    i_end = st.session_state.get("filter_end", const_date_range)
    if i_end - i_start < const_date_range:
        st.session_state.filter_end = i_end + 5
    else:
        st.session_state.filter_start = i_start + 5
        st.session_state.filter_end = i_end + 5

    cal_account_summary()


def main() -> None:
    global df, df1, df2
    # todo Settings
    construct_sidebar()

    st.header("Stock trading traing :moneybag: :dollar: :bar_chart:")

    with st.expander("How to Use This"):
        st.write(Path("README.md").read_text())

    # load data
    ticker = st.session_state.get("input_ticker", "MSFT")
    start = st.session_state.get("input_date_start", "2010-01-01")
    df = load_data(ticker, start=start)
    df["date"] = df.index

    with st.expander("Raw Dataframe"):
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination()
        gridOptions = gb.build()

        AgGrid(df, gridOptions=gridOptions, allow_unsafe_jscode=True)

    # cleaned data
    df1 = clean_data(df)
    with st.expander("Cleaned Data"):
        st.write(df1)

    # Display chart
    i_start = st.session_state.filter_start
    i_end = st.session_state.filter_end
    df2 = filter_data(df1, i_start, i_end)
    st.session_state.input_date_end = df2.index[-1].strftime("%Y-%m-%d")

    st.subheader("Charts")
    construct_chart_section(df2)

    with st.expander("logging"):
        st.text_area(label="", value=st.session_state.log_text, key="input_text_area1")


if __name__ == "__main__":
    st.set_page_config(
        "炒股训练",
        "📊",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()
