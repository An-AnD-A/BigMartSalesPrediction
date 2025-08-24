import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor, XGBRFRegressor
from catboost import CatBoostRegressor    
from lightgbm import LGBMRegressor

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance


from helpers.config import (train_data_path, test_data_path, output_base_path, 
                                processed_train_data_path, processed_test_data_path, 
                                item_metadata_path, outlet_metadata_path)
from helpers.DataReader import filereader
from utils.JsonImputer import JsonImputer
from utils.CustomImporter import BigMartFeatureEngineer
from utils.DataProcessing import detect_outliers

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def MultiEnsembleModel():

    # Loading the dataset
    train_df = filereader(train_data_path)
    X = train_df.drop(columns=['Item_Outlet_Sales'])
    y = train_df['Item_Outlet_Sales']

    # Data Processing
    custom_imputer = BigMartFeatureEngineer(year_ref=2025)

    one_hot_features = ['Item_Fat_Content',
                        'Outlet_Type',
                        'Item_Type_Combined',
                        ]
    
    label_features = [
        'MRP_Band',
        'Outlet_Identifier', 
        'Outlet_Size',
        'Outlet_Location_Type']
    
    numerical_features = [
        'Item_Weight',
        'Item_Visibility',
        'Years_Established',
        'MRP_squared',
        'Item_MRP']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('one_hot_cat', OneHotEncoder(handle_unknown="ignore"), one_hot_features),
        ('label_cat', OrdinalEncoder(), label_features)
    ])


    # Models

    lasso = Lasso(alpha=0.001, 
                  random_state=42,
                  max_iter=20000)
    

    xgb = XGBRegressor(
        objective="reg:squarederror", 
        n_estimators=1600, 
        learning_rate=0.001,
        max_depth=8, 
        subsample=0.8,  
        colsample_bytree=0.8, 
        reg_alpha=3.5,
        reg_lambda=4.5,
        random_state=42, 
        n_jobs=-1,
    )

    catboost = CatBoostRegressor(
        iterations=1950, 
        learning_rate=0.001, 
        depth=4,
        random_strength=0.5,
        loss_function="RMSE",
        l2_leaf_reg=2, 
        random_seed=42,
        verbose=0,
        subsample=0.3,
    )

    # Meta-learner: polynomial regression with Ridge
    meta_learner = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        Ridge(alpha=0.1)
    )

    # Stacking ensemble
    ensemble = StackingRegressor(
        estimators=[
            ('lasso', lasso),
            ('xgb', xgb),
            ('cat', catboost),
        ],
        final_estimator=meta_learner,
        n_jobs=-1
    )

    # Full Imputer
    model_pipeline = Pipeline([
        ('imputer', custom_imputer),
        ('preprocessor', preprocessor),
        ('model', ensemble)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = np.clip(model_pipeline.predict(X_test), 0, None)
    print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Test R²:", r2_score(y_test, y_pred))

    # Test Data Prediction
    test_df= filereader(test_data_path)

    test_preds = model_pipeline.predict(test_df)
    test_preds = np.maximum(0, test_preds)  # clip negatives to 0

    
    submission = pd.DataFrame({
        "Item_Identifier": test_df["Item_Identifier"],
        "Outlet_Identifier": test_df["Outlet_Identifier"],
        "Item_Outlet_Sales": test_preds
    })

    submission.to_csv(output_base_path / "final_submission" / "submission.csv", index=False)
    print("✅ Submission file saved as submission.csv")

if __name__ == '__main__':

    MultiEnsembleModel()