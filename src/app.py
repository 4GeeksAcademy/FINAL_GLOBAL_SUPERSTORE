from utils import db_connect
engine = db_connect()

# your code here
# [Tus imports anteriores permanecen igual]
import streamlit as st
from pickle import load
import pandas as pd
import joblib

# Carga de modelo y datos
model = load(open('../models/arima_0_0_1.pkl', 'rb'))
model_Storage = load(open('../models/Storage_arima_0_0_1.pkl', 'rb'))

ts_mensual = pd.read_csv('../data/processed/datos_procesados.csv', index_col=0, parse_dates=True)
ts_mensual_s = pd.read_csv('../data/processed/datos_procesados_Storage.csv', index_col=0, parse_dates=True)

st.title('StockSense')
st.markdown('Determinar la cantidad de stock en los próximos meses.')

valor         = st.slider('Cantidad de meses', min_value=1, max_value=12, step=1, value=4) 

if st.button("Predict"):
    # 1. Predicciones de desviaciones
    desviaciones = model.predict(valor)
    desv_storage = model_Storage.predict(valor) # Tu nuevo modelo

    # 2. Obtener el último valor base
    try:
        ultimo_valor_cantidad = ts_mensual['quantity'].iloc[-1]
        ultimo_storage = ts_mensual_s['quantity'].iloc[-1]
    except KeyError:
        ultimo_valor_cantidad = ts_mensual.iloc[-1, 0]

    # 3. Cálculo de cantidades reales (Acumulando desviaciones)
    cantidades_office = round(ultimo_valor_cantidad + desviaciones.cumsum(), 0)
    cantidades_storage = round(ultimo_storage + desv_storage.cumsum(), 0)

    # 4. Creación del DataFrame con ambas columnas
    df_predict = pd.DataFrame({
        'Stock Mensual office supplies totales': cantidades_office,
        'Stock Mensual Storage': cantidades_storage
    })

    # Mostrar resultados en tabla
    st.subheader("Predicción de Inventario")
    st.dataframe(df_predict)

    # Gráfico de líneas 
    st.line_chart(df_predict)