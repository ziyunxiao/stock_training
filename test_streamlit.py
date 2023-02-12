import cufflinks as cf
import numpy as np
import pandas as pd
import plotly
import streamlit as st

setattr(plotly.offline, "__PLOTLY_OFFLINE_INITIALIZED", True)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=["lat", "lon"]
)

st.map(map_data)


df = cf.datagen.ohlcv()
qf = cf.QuantFig(df, title="First Quant Figure", legend="top", name="GS")

qf.add_sma([10, 20], width=2, color=["green", "lightgreen"], legendgroup=True)
qf.add_rsi(periods=20, color="java")
qf.add_bollinger_bands(periods=20, boll_std=2, colors=["magenta", "grey"], fill=True)
qf.add_volume()
qf.add_macd()
fig = qf.iplot(asFigure=True)

st.plotly_chart(fig)
