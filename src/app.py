#from utils import db_connect
#engine = db_connect()

# your code here
# [Tus imports anteriores permanecen igual]
import joblib
import os
import pandas as pd
import pickle
from pickle import load
import streamlit as st

# 1. Configurar la ruta base del proyecto
# Esto apunta a la carpeta 'src', así que subimos un nivel para llegar a la raíz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Carga de modelos (Usando rutas seguras)
model_path = os.path.join(BASE_DIR, 'models', 'arima_0_0_1.pkl')
model_storage_path = os.path.join(BASE_DIR, 'models', 'Storage_arima_0_0_1.pkl')

model = load(open(model_path, 'rb'))
model_Storage = load(open(model_storage_path, 'rb'))

# 3. Carga de datos
ts_path = os.path.join(BASE_DIR, 'data', 'processed', 'datos_procesados.csv')
ts_s_path = os.path.join(BASE_DIR, 'data', 'processed', 'datos_procesados_Storage.csv')

ts_mensual = pd.read_csv(ts_path, index_col=0, parse_dates=True)
ts_mensual_s = pd.read_csv(ts_s_path, index_col=0, parse_dates=True)

st.title('StockSense')
st.markdown('Determinar la cantidad de stock en los próximos meses.')

valor = st.slider('Cantidad de meses', min_value=1, max_value=12, step=1, value=4) 

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