import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import traceback
import numpy as np

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from src.helpers.config import (train_data_path, test_data_path, output_base_path, 
                                processed_train_data_path, processed_test_data_path, 
                                item_metadata_path, outlet_metadata_path)
from src.helpers.DataReader import filereader
from src.model.JsonImputer import JsonImputer

from xgboost import XGBRegressor


def xgb_regressor_model():
    # Read the train dataset
    train_df = filereader(train_data_path)

    # Drop the target variable for processing
    feature_df = train_df.drop(columns=['Item_Outlet_Sales'])
    target_df = train_df['Item_Outlet_Sales']

    # Load metadata from JSON files
    with open(item_metadata_path, 'r') as f:
        item_metadata = json.load(f)

    with open(outlet_metadata_path, 'r') as f:
        outlet_metadata = json.load(f)

    # Custom Imputer
    custom_imputer = JsonImputer(item_metadata=item_metadata, outlet_metadata=outlet_metadata)

    # Define the categorical features to be one-hot encoded
    id_columns = ['Item_Identifier', 'Outlet_Identifier']
    categorical_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type','MRP_Band','Item_Category']
    numerical_features = ['Item_Weight','Item_Visibility','Item_MRP','Years']

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # drops id_columns automatically
    )

    # XGBRegressor with Poisson objective ensures non-negative predictions
    regressor = XGBRegressor(
        # objective="reg:poisson",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # Full pipeline
    model_pipeline = Pipeline(steps=[
        ('imputer', custom_imputer),
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, target_df, test_size=0.2, random_state=42
    )

    try:
        model_pipeline.fit(X_train, y_train)
    except Exception as e:
        traceback.print_exc()

    # Predictions
    y_pred = model_pipeline.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)  # safeguard (should already be ≥0)

    print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Test R²:", r2_score(y_test, y_pred))

    # Predict on test dataset
    test_df = filereader(test_data_path)
    predictions = model_pipeline.predict(test_df)
    predictions = np.clip(predictions, 0, None)  # safeguard again

    test_df['Item_Outlet_Sales'] = predictions
    test_df.to_csv(processed_test_data_path, index=False)
    print(f"Predictions saved to {processed_test_data_path}")


if __name__ == "__main__":
    xgb_regressor_model()
