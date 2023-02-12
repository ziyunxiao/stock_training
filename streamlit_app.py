import functools
from datetime import date
from pathlib import Path

import cufflinks as cf
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode

setattr(plotly.offline, "__PLOTLY_OFFLINE_INITIALIZED", True)


@st.cache_data
def load_data(ticker: str, start: str) -> pd.DataFrame:
    # todo change to load data from yfinance
    df = cf.datagen.ohlcv()
    return df


def get_chart(df: pd.DataFrame):
    qf = cf.QuantFig(df, title="First Quant Figure", legend="top", name="GS")

    qf.add_sma([10, 20], width=2, color=["green", "lightgreen"], legendgroup=True)
    qf.add_rsi(periods=20, color="java")
    qf.add_bollinger_bands(
        periods=20, boll_std=2, colors=["magenta", "grey"], fill=True
    )
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
    df = df.copy()
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


@st.cache_data
def filter_data(df: pd.DataFrame, start: date = None, end: date = None) -> pd.DataFrame:
    """
    Returns Dataframe with only accounts and symbols selected

    Args:
        df (pd.DataFrame): clean fidelity csv data, including account_name and symbol columns
        account_selections (list[str]): list of account names to include
        symbol_selections (list[str]): list of symbols to include

    Returns:
        pd.DataFrame: data only for the given accounts and symbols
    """
    df = df.copy()
    return df


def construct_settings():
    option = st.sidebar.selectbox(
        "Ticker",
        ("MSFT", "SQQQ", "TQQQ"),
        key="input_ticker",
    )

    d = st.sidebar.date_input("Start Date", date(2010, 1, 1), key="input_date_start")
    # st.sidebar.write('Your birthday is:', d)


def main() -> None:
    st.sidebar.subheader("Settings")
    # todo Settings
    construct_settings()

    st.header("Stock trading traing :moneybag: :dollar: :bar_chart:")

    with st.expander("How to Use This"):
        st.write(Path("README.md").read_text())

    df = load_data("MSFT", start="2010-01-01")
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
    st.subheader("Charts")
    st.write("Ticker:", st.session_state.input_ticker)
    st.write("Start:", st.session_state.input_date_start)

    fig = get_chart(df)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(
        "Fidelity Account View by Gerard Bentley",
        "📊",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()