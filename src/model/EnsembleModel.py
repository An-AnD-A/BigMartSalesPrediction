import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder, OrdinalEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor, XGBRFRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor

from src.helpers.config import (train_data_path, test_data_path, output_base_path, 
                                processed_train_data_path, processed_test_data_path, 
                                item_metadata_path, outlet_metadata_path)
from src.helpers.DataReader import filereader
from src.model.JsonImputer import JsonImputer
from src.Functions.DataProcessing import detect_outliers
import json

from xgboost import XGBRegressor

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def ensemble_model():
    # Load train data
    train_df = filereader(train_data_path)
    X = train_df.drop(columns=['Item_Outlet_Sales'])
    y = train_df['Item_Outlet_Sales']

    # Outlier processing
    upper_vis, lower_vis = detect_outliers(train_df, 'Item_Visibility')
    upper_sales, lower_sales = detect_outliers(train_df, 'Item_Outlet_Sales')

    train_df = train_df[(train_df['Item_Visibility'] > lower_vis) & (train_df['Item_Visibility'] < upper_vis)]
    train_df = train_df[(train_df['Item_Outlet_Sales'] > lower_sales) & (train_df['Item_Outlet_Sales'] < upper_sales)]

    # Load metadata from JSON files
    with open(item_metadata_path, 'r') as f:
        item_metadata = json.load(f)

    with open(outlet_metadata_path, 'r') as f:
        outlet_metadata = json.load(f)

    custom_imputer = JsonImputer(item_metadata=item_metadata, outlet_metadata=outlet_metadata)

    log_features = ['Item_MRP']
    one_hot_features = ['Item_Fat_Content','Outlet_Type','Item_Category']
    label_features = [ 'Item_Type','MRP_Band', 'Outlet_Identifier', 'Outlet_Size','Outlet_Location_Type']
    numerical_features = ['Item_Weight','Item_Visibility','Years']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('log_mrp', Pipeline([
            ('log', FunctionTransformer(np.log1p, validate=False)),
            ('scale', StandardScaler())
        ]), log_features),
        ('one_hot_cat', OneHotEncoder(handle_unknown="ignore"), one_hot_features),
        ('label_cat', OrdinalEncoder(), label_features)
    ])

    # Base models
    lasso = Lasso(alpha=0.1, 
                  random_state=42,
                  max_iter=15000)
                  
    xgb = XGBRegressor(
        objective="reg:squarederror", 
        n_estimators=800, 
        learning_rate=0.02,
        max_depth=6 , 
        subsample=0.5,  
        colsample_bytree=0.5, 
        reg_alpha=3,
        reg_lambda=4,
        random_state=42, 
        n_jobs=-1
    )

    xgbrf = XGBRFRegressor(
        objective="reg:squarederror",
        n_estimators=600,         # usually fewer needed than boosting
        learning_rate=0.05,       # slightly higher since no boosting correction
        max_depth=8,
        subsample=0.8,
        colsample_bynode=0.8,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1
    )

    catboost = CatBoostRegressor(
        iterations=1500, 
        learning_rate=0.02, 
        depth=6,
        loss_function="RMSE",
        l2_leaf_reg=10, 
        random_seed=42,
        verbose=0
    )

    # Meta-learner: polynomial regression with Ridge
    meta_learner = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        Ridge(alpha=2) # 2 -> 3
    )


    # Stacking ensemble
    ensemble = StackingRegressor(
        estimators=[
            ("lasso", lasso), 
            ('xgbrf', xgbrf),
            ("xgb", xgb),
            ('cat', catboost)],
        # final_estimator=Lasso(alpha=0.0005, max_iter=10000), # meta-learner
        # final_estimator=Ridge(alpha=1.0),
        final_estimator=meta_learner,
        n_jobs=-1
    )

    # Full pipeline
    model_pipeline = Pipeline([
        ('imputer', custom_imputer),
        ('preprocessor', preprocessor),
        ('model', ensemble),
        # ('model', TransformedTargetRegressor(
        #     regressor=ensemble,
        #     func=np.log1p,       # log(1 + y)
        #     inverse_func=np.expm1 # exp(y) - 1
        # ))
    ])

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_pipeline, X, y, cv=kf, scoring=make_scorer(rmse, greater_is_better=False))
    print("CV RMSE scores:", -cv_scores)
    print("Mean CV RMSE:", -cv_scores.mean())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = np.clip(model_pipeline.predict(X_test), 0, None)
    print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Test R²:", r2_score(y_test, y_pred))

    # # Blended predictions (processed pipeline applied to X_train/X_test)
    # X_train_proc = model_pipeline.named_steps['preprocessor'].fit_transform(
    #     model_pipeline.named_steps['imputer'].transform(X_train)
    # )
    # X_test_proc = model_pipeline.named_steps['preprocessor'].transform(
    #     model_pipeline.named_steps['imputer'].transform(X_test)
    # )

    # lasso.fit(X_train_proc,y_train)
    # xgb.fit(X_train_proc, y_train)
    # catboost.fit(X_train_proc, y_train)
    # lasso_preds = np.clip(lasso.predict(X_test_proc), 0, None)
    # xgb_preds = np.clip(xgb.predict(X_test_proc), 0, None)
    # cat_preds = np.clip(catboost.predict(X_test_proc), 0, None)

    # blended_preds = 0.2 * xgb_preds + 0.7 * cat_preds + 0.1 * lasso_preds
    # print("Blended Test RMSE:", np.sqrt(mean_squared_error(y_test, blended_preds)))

    # Final prediction
    test_df = filereader(test_data_path)
    test_proc = model_pipeline.named_steps['preprocessor'].transform(
        model_pipeline.named_steps['imputer'].transform(test_df)
    )
    preds = np.clip(model_pipeline.named_steps['model'].predict(test_proc), 0, None)
    # preds = np.clip(model_pipeline.predict(test_df), 0, None)
    test_df["Item_Outlet_Sales"] = preds
    test_df.to_csv(processed_test_data_path, index=False)
    print(f"Predictions saved to {processed_test_data_path}")

    return