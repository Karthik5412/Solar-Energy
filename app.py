from transformers import Preprocessing, Encoding
import pandas as pd
import numpy as np
import streamlit as st
from datetime import timedelta, combine, datetime
import joblib
import requests

date = st.sidebar.date_input('Enter date :')
time = st.sidebar.time_input('Enter time :')

st_date = combine(date,time) 
en_date = st_date + timedelta(hours= 1 )

st_date_str = datetime(st_date).strftime('%Y-%m-%dT%H:%M')
en_date_str = datetime(en_date).strftime('%Y-%m-%dT%H:%M')

base_url = 'https://api.open-meteo.com/v1/forecast?latitude=17.3850&longitude=78.4867&hourly=temperature_2m,direct_radiation,diffuse_radiation'
date_range = f'&start_hour={st_date_str}&end_hour={en_date_str}'
time_zone = r'&timezone=Asia%2FKolkata'

url = base_url+date_range + time_zone

response = requests.get(url).json()

NOCT = 45.0
current = response['hourly']

temp = current['temperature_2m']
irr = current['direct_radiation'] + current['diffuse_radiation']

module = temp + ((NOCT - 20) / 800) * irr

pipeline = joblib.load(r'pipe.plk')



data = {
    'DATE_TIME' : [current['time']] ,
    'AMBIENT_TEMPERATURE' : [temp] ,
    'MODULE_TEMPERATURE' : [module],
    'IRRADIATION' : [irr]
}

df = pd.DataFrame(data)

st.success(df)
st.success(pipeline.predict(df))