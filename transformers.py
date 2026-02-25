import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessing(BaseEstimator, TransformerMixin):
    def fit(self,x, y=None):
        return self
    
    def transform(self,x):
        
        # Filling data with farword filling
        x = x.fillna(method = 'ffill')
        x = x.copy()
        # Feature extraction
        x['DATE_TIME'] = pd.to_datetime(x['DATE_TIME'])
        
        x['Month'] = x['DATE_TIME'].dt.month
        x['Day'] = x['DATE_TIME'].dt.day
        x['Day_of_week'] = x['DATE_TIME'].dt.weekday

        x['Hour'] = x['DATE_TIME'].dt.hour
        x['Minute'] = x['DATE_TIME'].dt.minute

        
        
        return x.drop(columns= ['DATE_TIME'])
    
    
class Encoding(BaseEstimator, TransformerMixin):
    def fit(self,x, y=None):
        return self
    
    def transform (self, x):
        x_encoded = x.copy()
        
        # Time encoding
        x_encoded['hour_sin'] = np.sin(2 * np.pi * x_encoded['Hour'] / 24)
        x_encoded['hour_cos'] = np.cos(2 * np.pi * x_encoded['Hour'] / 24)

        x_encoded['minute_sin'] = np.sin(2 * np.pi * x_encoded['Minute'] / 60)
        x_encoded['minute_cos'] = np.cos(2 * np.pi * x_encoded['Minute'] / 60)
        
        # Date encoding
        x_encoded['month_sin'] = np.sin(2 * np.pi * (x_encoded['Month'] -1) / 12)
        x_encoded['month_cos'] = np.cos(2 * np.pi * (x_encoded['Month'] -1) / 12)

        x_encoded['day_sin'] = np.sin(2 * np.pi * x_encoded['Day']- 1/ 31 )
        x_encoded['day_cos'] = np.cos(2 * np.pi * x_encoded['Day'] -1/ 31 )

        x_encoded['week_day_sin'] = np.sin(2 * np.pi * x_encoded['Day_of_week'] / 7)
        x_encoded['week_day_cos'] = np.cos(2 * np.pi * x_encoded['Day_of_week'] / 7)
        
        drop_cols = ['Hour', 'Minute','Month', 'Day', 'Day_of_week' ]
        
        
        return x_encoded.drop(columns= drop_cols)