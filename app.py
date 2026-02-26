from transformers import Preprocessing, Encoding
import pandas as pd
import numpy as np
import streamlit as st
from datetime import timedelta,  datetime
import joblib
import requests

date = st.sidebar.date_input('Enter date :')
time = st.sidebar.time_input('Enter time :').strftime('%H')

st_date = date
en_date = st_date + timedelta(days= 1 )


base_url = 'https://api.open-meteo.com/v1/forecast?latitude=17.3850&longitude=78.4867&hourly=temperature_2m,direct_radiation,diffuse_radiation'
date_range = f'&start_date={st_date}&end_date={en_date}'
time_zone = r'&timezone=Asia%2FKolkata'

url = base_url+date_range + time_zone

response = requests.get(url).json()

NOCT = 45.0
current = response['hourly']

temp = current['temperature_2m'][int(time)]

irr = current['direct_radiation'][int(time)] + current['diffuse_radiation'][int(time)]

module = temp + ((NOCT - 20) / 800) * irr


pipeline = joblib.load(r'pipe.plk')


dt_format = pd.to_datetime(current['time'][int(time)]).strftime('%Y-%m-%d %H:%M:%S')


data = {
    'DATE_TIME' : [dt_format] ,
    'AMBIENT_TEMPERATURE' : [temp] ,
    'MODULE_TEMPERATURE' : [module],
    'IRRADIATION' : [irr]
}

df = pd.DataFrame(data)

pred = pipeline.predict(df)[0]
pred = list(pred)
st.success(pred[0 :])
# st.success(f'DC_POWER : {pred[0]} DAY_DC_POWER : {pred[1]}')