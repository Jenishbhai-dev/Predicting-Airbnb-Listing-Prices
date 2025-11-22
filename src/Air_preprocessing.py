import pandas as pd
import numpy as np
import boto3
import pandas as pd
from io import StringIO
import logging
from dotenv import load_dotenv
import os
class AirbnbPreprocessor:
    def handle_missing_values(self, df):
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        return df

    def remove_outliers_iqr(self, df, columns, factor=1.5):
        df = df.copy()
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - factor * IQR) & (df[col] <= Q3 + factor * IQR)]
        return df
