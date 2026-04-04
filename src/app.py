from utils import db_connect
engine = db_connect()

# your code here
import streamlit as st
from pickle import load
import pandas as pd

model = load(open('/workspaces/FINAL_GLOBAL_SUPERSTORE/models/arima_0_0_1.pkl', 'rb'))

st.title('StockSense')
st.markdown('Determinar la cantidad de stock en los proximos meses.')

valor = st.slider('Cantidad de meses', min_value = 0, max_value=12, step= 1)


if st.button("Predict"):
    prediction = model.predict(valor)
    data = pd.DataFrame(prediction, columns=['Desviacion'])
    st.subheader(f"Resultados para los próximos {valor} meses")
    st.dataframe(data)
