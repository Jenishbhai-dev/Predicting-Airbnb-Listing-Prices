import pandas as pd
import numpy as np

class FeatureEngineer:
    def create_text_features(self, df, text_columns):
        df = df.copy()
        for col in text_columns:
            if col in df.columns:
                df[f'{col}_length'] = df[col].astype(str).str.len()
                df[f'{col}_word_count'] = df[col].astype(str).str.split().str.len()
        return df

    def create_availability_features(self, df):
        df = df.copy()
        avail_cols = [col for col in df.columns if 'availability' in col.lower()]
        if avail_cols:
            df['total_availability'] = df[avail_cols].sum(axis=1)
            if 'availability_365' in df.columns:
                df['availability_ratio'] = df['availability_365'] / 365
        return df

    def create_review_features(self, df):
        df = df.copy()
        if 'number_of_reviews' in df.columns and 'reviews_per_month' in df.columns:
            df['review_engagement'] = df['number_of_reviews'] * df['reviews_per_month']
            df['reviews_per_month'].fillna(0, inplace=True)
            df['is_active'] = (df['reviews_per_month'] > 0).astype(int)
        return df

    def create_location_features(self, df):
        df = df.copy()
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['distance_from_center'] = np.sqrt(df['latitude']**2 + df['longitude']**2)
        if 'neighbourhood' in df.columns:
            neighbourhood_counts = df['neighbourhood'].value_counts()
            df['neighbourhood_density'] = df['neighbourhood'].map(neighbourhood_counts)
        return df

    def create_price_features(self, df):
        df = df.copy()
        if 'price' in df.columns and 'bedrooms' in df.columns:
            df['price_per_bedroom'] = df['price'] / (df['bedrooms'] + 1)
        if 'price' in df.columns and 'accommodates' in df.columns:
            df['price_per_person'] = df['price'] / (df['accommodates'] + 1)
        return df

    def create_amenity_features(self, df):
        df = df.copy()
        if 'amenities' in df.columns:
            df['amenities_count'] = df['amenities'].astype(str).str.count(',') + 1
            for amenity in ['wifi', 'pool', 'gym', 'parking', 'kitchen', 'washer', 'dryer']:
                df[f'has_{amenity}'] = df['amenities'].astype(str).str.lower().str.contains(amenity).astype(int)
        return df

    def create_host_features(self, df):
        df = df.copy()
        if 'host_is_superhost' in df.columns:
            df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
        if 'calculated_host_listings_count' in df.columns:
            df['is_professional_host'] = (df['calculated_host_listings_count'] > 5).astype(int)
        return df

    def create_all_features(self, df):
        df = self.create_text_features(df, ['name'])
        df = self.create_availability_features(df)
        df = self.create_review_features(df)
        df = self.create_location_features(df)
        df = self.create_price_features(df)
        df = self.create_amenity_features(df)
        df = self.create_host_features(df)
        return df
