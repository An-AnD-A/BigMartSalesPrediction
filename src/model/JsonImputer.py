import json
import pandas as pd
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, TransformerMixin

from src.helpers.config import train_data_path, test_data_path, output_base_path, item_metadata_path, outlet_metadata_path
from src.helpers.DataReader import filereader


class JsonImputer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 item_metadata : dict =None,
                 outlet_metadata : dict = None):
        
        super().__init__()
        self.item_metadata = item_metadata
        self.outlet_metadata = outlet_metadata
        self.item_visibility_median_details = None
        self.item_visibility_mean_details = None
        self.global_visibility_median_ = None

    def fit(self, X, y=None):
        
        X_copy = X

        vis = X_copy['Item_Visibility'].replace(0,np.nan)
        self.item_visibility_median_details = X_copy.assign(Item_Visibility=vis).groupby('Item_Identifier')[
            'Item_Visibility'
        ].median()
        self.item_visibility_mean_details = X_copy.assign(Item_Visibility=vis).groupby('Item_Identifier')[
            'Item_Visibility'
        ].mean()

        self.global_visibility_median_ = vis.median()

        return self
    
    def transform(self, X):

        # Map the item metadata

        X_copy = X.copy()

        for idx, row in X_copy.iterrows():
            item_id = row['Item_Identifier']
            outlet_id = row['Outlet_Identifier']

            if item_id in self.item_metadata:
                for k, v in self.item_metadata[item_id].items():
                    if pd.isna(row[k]):
                        X_copy.at[idx, k] = v
                    

            if outlet_id in self.outlet_metadata:
                for k, v in self.outlet_metadata[outlet_id].items():
                    if pd.isna(row[k]):
                        X_copy.at[idx, k] = v

        # MRP Band
        try:
            X_copy["MRP_Band"] = pd.qcut(X_copy["Item_MRP"], q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
        except Exception:
            X_copy["MRP_Band"] = pd.cut(X_copy["Item_MRP"], bins=4, labels=["Q1","Q2","Q3","Q4"], include_lowest=True)

        #Reference Year
        X_copy['Years'] = 2025 - X_copy['Outlet_Establishment_Year']

        # Imputation for visibility

        X_copy["Item_Visibility"] = X_copy["Item_Visibility"].replace(0, np.nan)
        X_copy["Item_Visibility"] = X_copy.apply(
            lambda r: r["Item_Visibility"]
            if pd.notnull(r["Item_Visibility"]) else self.item_visibility_median_details.get(r["Item_Identifier"], np.nan),
            axis=1,
        )
        X_copy["Item_Visibility"].fillna(self.global_visibility_median_, inplace=True)

        # Item Category identification

        X_copy['Item_Category'] = X_copy['Item_Identifier'].str[:2]

        return X_copy
    
# Check if the custom transformer is working correctly
if __name__ == "__main__":
    # Load metadata from JSON files
    with open(item_metadata_path, 'r') as f:
        item_metadata = json.load(f)
    
    with open(outlet_metadata_path, 'r') as f:
        outlet_metadata = json.load(f)

    # Create a sample DataFrame
    df = pd.read_csv(test_data_path)

    # Initialize and test the JsonImputer
    imputer = JsonImputer(item_metadata=item_metadata, outlet_metadata=outlet_metadata)
    imputer.fit(df)  # <-- Add this line
    transformed_df = imputer.transform(df)

    transformed_df.info()



