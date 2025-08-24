import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import traceback
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error


from src.helpers.config import (train_data_path, test_data_path, output_base_path, 
                                processed_train_data_path, processed_test_data_path, 
                                item_metadata_path, outlet_metadata_path)
from src.helpers.DataReader import filereader
from utils.JsonImputer import JsonImputer

def RFregressor_model():
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
    categorical_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type','MRP_Band']
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

    # Log transform wrapper for y
    # def log_transform(y):
    #     return np.log1p(y)

    # def inverse_log_transform(y):
    #     return np.expm1(y)

    # Full pipeline
    model_pipeline = Pipeline(steps=[
        ('imputer', custom_imputer),
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, target_df, test_size=0.2, random_state=42
    )

    # y_train_log = log_transform(y_train)

    # # Parameter grid for tuning
    # param_grid = {
    #     'regressor__n_estimators': [100, 300, 500],
    #     'regressor__max_depth': [10, 20, None],
    #     'regressor__min_samples_split': [2, 5, 10],
    #     'regressor__min_samples_leaf': [1, 2, 5],
    #     'regressor__max_features': ['sqrt', 'log2']
    # }

    # # GridSearchCV
    # grid_search = GridSearchCV(
    #     model_pipeline,
    #     param_grid,
    #     cv=3,
    #     scoring='neg_root_mean_squared_error',
    #     n_jobs=-1,
    #     verbose=2
    # )

    try:
        model_pipeline.fit(X_train, y_train)
    except Exception as e:
        traceback.print_exc()
    # print(grid_search)

    # print("Best params:", grid_search.best_params_)
    # print("Best CV RMSE:", -grid_search.best_score_)

    # Evaluate on test set
    # best_model = grid_search.best_estimator_
    y_pred_log = model_pipeline.predict(X_test)
    y_pred = y_pred_log

    print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Test R²:", r2_score(y_test, y_pred)) 

    # Make the actual predictions on the test dataset
    test_df = filereader(test_data_path)

    predictions = model_pipeline.predict(test_df)
    test_df['Item_Outlet_Sales'] = predictions
    test_df.to_csv(processed_test_data_path, index=False)
    print(f"Predictions saved to {processed_test_data_path}")


if __name__ == "__main__":

    RFregressor_model()





    