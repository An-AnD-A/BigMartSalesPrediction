import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from utils.CustomImporter import BigMartFeatureEngineer
from sklearn.compose import TransformedTargetRegressor
from helpers.config import (train_data_path, test_data_path, output_base_path, 
                                processed_train_data_path, processed_test_data_path, 
                                item_metadata_path, outlet_metadata_path)
from helpers.DataReader import filereader


# Load data
train = pd.read_csv(train_data_path)
test = pd.read_csv(test_data_path)

X = train.drop(columns=["Item_Outlet_Sales"])
y = train["Item_Outlet_Sales"]

# Build pipeline
pipeline = Pipeline([
    ("features", BigMartFeatureEngineer(year_ref=2013)),
    ("model", TransformedTargetRegressor(
        regressor=XGBRegressor(
            n_estimators=1500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2,
            reg_alpha=0.5,
            random_state=42,
            n_jobs=-1
        ),
        func=np.log1p,
        inverse_func=np.expm1
    ))
])

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit
pipeline.fit(X_train, y_train)

# Validate
y_pred = pipeline.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.2f}")

test_preds = pipeline.predict(test)
test_preds = np.maximum(0, test_preds)  # clip negatives to 0

# ---------------------------
# Create Submission File
# ---------------------------
submission = pd.DataFrame({
    "Item_Identifier": test["Item_Identifier"],
    "Outlet_Identifier": test["Outlet_Identifier"],
    "Item_Outlet_Sales": test_preds
})

submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved as submission.csv")