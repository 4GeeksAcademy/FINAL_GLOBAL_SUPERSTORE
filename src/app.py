from utils import db_connect
engine = db_connect()

# your code here
# [Tus imports anteriores permanecen igual]
import streamlit as st
from pickle import load
import pandas as pd
import joblib

# Carga de modelo y datos
model = load(open('/workspaces/FINAL_GLOBAL_SUPERSTORE/models/arima_0_0_1.pkl', 'rb'))

ts_mensual = pd.read_csv('/workspaces/FINAL_GLOBAL_SUPERSTORE/data/processed/datos_procesados.csv', index_col=0, parse_dates=True)


st.title('StockSense')
st.markdown('Determinar la cantidad de stock en los próximos meses.')

valor = st.slider('Cantidad de meses', min_value=1, max_value=12, step=1, value=4) # <- CAMBIO: min_value=1 para evitar predicción de 0 meses

if st.button("Predict"):
    
    desviaciones = model.predict(valor)
    try:
        ultimo_valor_cantidad = ts_mensual['quantity'].iloc[-1]
    except KeyError:
        
        ultimo_valor_cantidad = ts_mensual.iloc[-1, 0] 

    desviaciones_acumuladas = desviaciones.cumsum()
    cantidades_reales = round(ultimo_valor_cantidad + desviaciones_acumuladas,0)

    df_predict = pd.DataFrame(cantidades_reales)
    df_predict.columns = ['Stock Mensual office supplies'] 

    # Mostrar resultados en tabla
    st.subheader("Predicción de Inventario")
    st.dataframe(df_predict)

    # Gráfico de líneas 
    st.line_chart(df_predict)