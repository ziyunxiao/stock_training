import cufflinks as cf
import numpy as np
import pandas as pd
import streamlit as st

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=["lat", "lon"]
)

st.map(map_data)
