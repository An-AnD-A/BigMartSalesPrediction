import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class BigMartFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean and engineer features for Big Mart Sales Prediction.
    """

    def __init__(self, year_ref=2020):
        self.year_ref = year_ref
        self.label_encoders = {}
        self.cat_cols = ['Item_Fat_Content','Outlet_Identifier',
                         'Outlet_Size','Outlet_Location_Type',
                         'Outlet_Type','Item_Type_Combined', 'Item_Type']
        self.item_vis_mean_ = None
        self.global_vis_mean_ = None

    def fit(self, X, y=None):
        df = X.copy()

        # ---- Handle Missing ----
        df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
        df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])

        # ---- Consistency Fix ----
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(
            {'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'}
        )

        # ---- Feature Engineering ----
        df['Years_Established'] = self.year_ref - df['Outlet_Establishment_Year']
        df['Item_Visibility'] = df['Item_Visibility'].replace(0, df['Item_Visibility'].mean())
        df['Item_Type_Combined'] = df['Item_Identifier'].str[:2].map(
            {'FD':'Food','NC':'Non-Consumable','DR':'Drinks'}
        )
        df.loc[df['Item_Type_Combined']=='Non-Consumable','Item_Fat_Content'] = 'Non-Edible'

        df["MRP_squared"] = df["Item_MRP"] ** 2
        # df["MRP_x_Years"] = df["Item_MRP"] * df["Years_Established"]

        self.item_vis_mean_ = df.groupby('Item_Identifier')['Item_Visibility'].mean()
        self.global_vis_mean_ = df['Item_Visibility'].mean()


        return self

    def transform(self, X):
        df = X.copy()

        # ---- Handle Missing ----
        df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
        df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])

        # ---- Consistency Fix ----
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(
            {'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'}
        )

        # ---- Feature Engineering ----
        df['Years_Established'] = self.year_ref - df['Outlet_Establishment_Year']
        df['Item_Visibility'] = df['Item_Visibility'].replace(0, df['Item_Visibility'].mean())
        df['Item_Type_Combined'] = df['Item_Identifier'].str[:2].map(
            {'FD':'Food','NC':'Non-Consumable','DR':'Drinks'}
        )
        df.loc[df['Item_Type_Combined']=='Non-Consumable','Item_Fat_Content'] = 'Non-Edible'

        df["MRP_squared"] = df["Item_MRP"] ** 2
        # df["MRP_x_Years"] = df["Item_MRP"] * df["Years_Established"]

        # stable mean ratio
        vis_mean = df['Item_Identifier'].map(self.item_vis_mean_).fillna(self.global_vis_mean_)
        df['Item_Visibility_MeanRatio'] = df['Item_Visibility'] / vis_mean

        # Create price Band
        try:
            df["MRP_Band"] = pd.qcut(df["Item_MRP"], q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
        except Exception:
            df["MRP_Band"] = pd.cut(df["Item_MRP"], bins=4, labels=["Q1","Q2","Q3","Q4"], include_lowest=True)


        # ---- Drop unused columns ----
        df = df.drop(columns=['Item_Identifier','Outlet_Establishment_Year','Item_Type'], errors='ignore')

        return df