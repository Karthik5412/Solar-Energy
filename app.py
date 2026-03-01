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
            [data-testid = 'stSidebar'] {
                background : linear-gradient(135deg, #fceABB 0%, #f8b500 100%);
                color : #CD5B45;
            }
            
            
            </style>
            ''',unsafe_allow_html=True
)

st.title('Solar Production Predictor', text_alignment= 'center')

cities = {
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Kochi": {"lat": 9.9312, "lon": 76.2673},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794},
    "Patna": {"lat": 25.5941, "lon": 85.1376}
}

city = st.sidebar.selectbox('Enter City: ', cities.keys())
date = st.sidebar.date_input('Enter date :')
time = st.sidebar.time_input('Enter time :').strftime('%H')
btn = st.sidebar.button('Predict')

st_date = date
en_date = st_date + timedelta(days= 1 )
loc = cities[city]

try :
    base_url = 'https://api.open-meteo.com/v1/forecast?'
    loc_url = f'latitude={loc['lat']}&longitude={loc['lon']}&hourly=temperature_2m,direct_radiation,diffuse_radiation'
    date_range = f'&start_date={st_date}&end_date={en_date}'
    time_zone = r'&timezone=Asia%2FKolkata'

    url = base_url+loc_url+date_range + time_zone

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
        
    if pred[0] < 4500 :
        dc_color = '#4169E1'
        
    elif pred[0] > 4500 and pred[0] < 13000 :
        dc_color = '#2E8B57'
    else :
        dc_color = 'coral'   
    if btn :     
        st.balloons()
        st.markdown(f"### <span style='color: {dc_color};'><center>Avg DC Power: {round(pred[0],2)} w</center></span>", 
                    unsafe_allow_html=True)
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
                        'color' : text_color,
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
                    'axis' : {'range' : [0, 55], 'tickwidth' :2, 'tickcolor' : 'red', 'tickfont' : 
                        {'color' : bar_color, 'size' : 14}},
                    'bar' : {'color' : bar_color},
                    'bgcolor' : 'black',
                    'borderwidth' : 3,
                    'bordercolor' : border_color,
                    'threshold': {
                    'line': {'color': text_color, 'width': 4},
                    'thickness': 0.75,
                    'value': temp
                }
                    }
                )
            )
            
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with dia2:
            
            if pred[1] < 2400 :
                bar_color = 'skyblue'
                text_color = '#1A237E'
                border_color = 'blue'
            elif pred[1]  > 4500 and pred[1]  < 7200 :
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
                        'text' : '<b>DAY DC POWER</b>',
                        'font' : {
                        'color' : text_color,
                        'size'  : 24
                        }
                        } ,
                    value= pred[1],
                    domain= {'x' : [0,1], 'y' : [0,1]},
                    number={
                        'font' : {'color' : text_color},
                        'suffix' : ' w'
                        },
                    gauge={
                    'axis' : {'range' : [0, 7200], 'tickwidth' :2, 'tickcolor' : 'red', 'tickfont' : 
                        {'color' : bar_color, 'size' : 14}
                        },
                    'bar' : {'color' : bar_color},
                    'bgcolor' : 'black',
                    'borderwidth' : 3,
                    'bordercolor' : border_color,
                    'threshold': {
                    'line': {'color': text_color, 'width': 4},
                    'thickness': 0.75,
                    'value': pred[1]
                }
                    }
                )
            )
            
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
    
except Exception :
    st.success('You entered Date out of the range !')