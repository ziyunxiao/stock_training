import functools
import json
import logging
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
if "input_init_funds" not in st.session_state:
    st.session_state["input_init_funds"] = 100000

if "account_summary" not in st.session_state:
    st.session_state["account_summary"] = {
        "total": st.session_state["input_init_funds"],
        "total_delta": 0,
        "total_percent": 100,
    }

if "log_text" not in st.session_state:
    st.session_state["log_text"] = ""

if "trade_price" not in st.session_state:
    st.session_state["trade_price"] = "next day"

if "stop_price" not in st.session_state:
    st.session_state["stop_price"] = "10%"

if "current_holding_share" not in st.session_state:
    st.session_state["current_holding_share"] = 0
    st.session_state["current_holding_value"] = 0

if "filter_start" not in st.session_state:
    st.session_state["filter_start"] = 0
    st.session_state["filter_end"] = 60


@st.cache_data
def load_data(ticker: str, start: str) -> pd.DataFrame:
    # todo change to load data from yfinance
    msft = yf.Ticker(ticker)
    # get all stock info (slow)
    info = msft.info
    json.dump(info.info, open(f"./data/{ticker}_info.json", "w"), indent=4)

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


def filter_data(df: pd.DataFrame, start: int = 0, end: int = -1) -> pd.DataFrame:
    """
    Returns Dataframe with only accounts and symbols selected

    Args:
        df (pd.DataFrame): clean fidelity csv data, including account_name and symbol columns
        account_selections (list[str]): list of account names to include
        symbol_selections (list[str]): list of symbols to include

    Returns:
        pd.DataFrame: data only for the given accounts and symbols
    """
    i_start = st.session_state.get("filter_start", 0)
    i_end = st.session_state.get("filter_end", 60)
    df = df.iloc[i_start:i_end]
    return df


def update_init_fund():
    st.session_state["account_summary"] = {
        "total": st.session_state["input_init_funds"],
        "total_delta": 0,
        "total_percent": 100,
    }


def construct_sidebar():
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

    st.sidebar.subheader("Account Summary")
    st.sidebar.metric(
        "Total",
        f"${st.session_state.account_summary['total']:,d}",
        f"{st.session_state.account_summary['total_delta']:.2f}",
    )

    st.sidebar.metric(
        "Percentage",
        f"${st.session_state.account_summary['total_percent']}%",
    )


def construct_chart_section(df: pd.DataFrame):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Ticker:", st.session_state.input_ticker)

    with col2:
        st.write("Current Date:", st.session_state.input_date_end)

    with col3:
        st.write("Holding Value:", st.session_state.current_holding_value)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Next Day", key="btn_next_day", on_click=btn_next_day_click)

    with col2:
        st.button("Buy", key="btn_buy")

    with col3:
        st.button("Sell", key="btn_sell")

    fig = get_chart(df)
    st.plotly_chart(fig, use_container_width=True)


def btn_next_day_click():
    i_start = st.session_state.get("filter_start", 0)
    i_end = st.session_state.get("filter_end", 60)
    st.session_state.filter_start = i_start + 1
    st.session_state.filter_end = i_end + 1


def main() -> None:
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
        # gb.configure_columns(
        #     (
        #         "open",
        #         "high",
        #         "low",
        #         "close",
        #         "volume",
        #     )
        # )
        gb.configure_pagination()
        # gb.configure_columns(("account_name", "symbol"), pinned=True)
        gridOptions = gb.build()

        AgGrid(df, gridOptions=gridOptions, allow_unsafe_jscode=True)

    # cleaned data
    df = clean_data(df)
    with st.expander("Cleaned Data"):
        st.write(df)

    # Display chart
    df = filter_data(df)
    st.session_state.input_date_end = df.index[-1].strftime("%Y-%m-%d")
    st.subheader("Charts")
    construct_chart_section(df)

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
