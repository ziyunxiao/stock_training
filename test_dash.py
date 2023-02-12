import cufflinks as cf
import pandas as pd
import plotly
import plotly.express as px
from dash import Dash, dcc, html

setattr(plotly.offline, "__PLOTLY_OFFLINE_INITIALIZED", True)

df = cf.datagen.ohlcv()
qf = cf.QuantFig(df, title="First Quant Figure", legend="top", name="GS")

qf.add_sma([10, 20], width=2, color=["green", "lightgreen"], legendgroup=True)
# qf.add_rsi(periods=20, color="java")
qf.add_bollinger_bands(periods=20, boll_std=2, colors=["magenta", "grey"], fill=True)
qf.add_volume()
# qf.add_macd()
figure = qf.iplot(asFigure=True)

# figure = df.iplot(kind='scatter', asFigure=True)

app = Dash()
app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [html.H3("charts"), dcc.Graph(id="g1", figure=figure)],
                    className="six columns",
                )
            ]
        )
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
