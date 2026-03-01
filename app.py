from transformers import Preprocessing, Encoding
import pandas as pd
import numpy as np
import streamlit as st
from datetime import timedelta,  datetime
import joblib
import requests
import plotly.graph_objects as go 

st.set_page_config(page_title='Solar', page_icon='🌞', layout='wide')

st.markdown( '''<style>
            .stApp {
                background : linear-gradient(135deg, #fceABB 0%, #f8b500 100%);
                color : #CD5B45;
            }
            
            stPlotlyChart{
                background-color : tranparent ! important;
                border-radius : 15px ;
                box-shadow : 0 4px 6px rgba(0,0,0,0.1);
                padding : 10px;
            }
            
            </style>
            ''',unsafe_allow_html=True
)

st.title('Solar Production Predictor', text_alignment= 'center')

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

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Temperatur ')
    st.subheader(f'{temp} °C' )
with col2 :
    st.subheader('Irradiation ')
    st.subheader(f'{irr} W/m²' )
with col3 :
    st.subheader('Module Temperature ')
    st.subheader(f'{module} °C' )
    
dia1,dia2 = st.columns(2)

with dia1:
    
    if temp < 15 :
        bar_color = 'skyblue'
        text_color = '#1A237E'
        border_color = 'blue'
    elif temp > 15 and temp < 30 :
        bar_color = 'springgreen'
        text_color = '#2E7D32'
        border_color = 'green'
    else :
        bar_color = 'orange'
        text_color = '#B71C1C'
        border_color = 'red'
    
    fig = go.Figure(
        go.Indicator(
            mode='gauge+number',
            title= {
                'text' : '<b>TEMPERATURE</b>',
                'font' : {
                'color' : '#CD5B45',
                'size'  : 24
                }
                } ,
            value= temp,
            domain= {'x' : [0,1], 'y' : [0,1]},
            number={
                'font' : {'color' : text_color},
                'suffix' : ' °C'
                },
            gauge={
            'axis' : {'range' : [0, 55], 'tickwidth' :2, 'tickcolor' : 'red', 'tickfont' : {'color' : 'red', 'size' : 14}},
            'bar' : {'color' : bar_color},
            'bgcolor' : 'black',
            'borderwidth' : 3,
            'bordercolor' : border_color,
            'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 90
        }
            }
        )
    )
    
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    

# st.success()
# st.success()
    
# except Exception :
#     st.success('You entered Date out of the range !')