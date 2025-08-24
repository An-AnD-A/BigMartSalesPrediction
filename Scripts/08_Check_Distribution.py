import matplotlib.pyplot as plt
import pandas as pd

from helpers.config import train_data_path, test_data_path
from helpers.DataReader import filereader

train_df = filereader(train_data_path)
test_df = filereader(test_data_path)
submission_df = pd.read_csv(r'C:\Users\anand\Projects\BigMartSalesPrediction\submission.csv')

train_df["MRP_Band"] = pd.qcut(train_df["Item_MRP"], q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")

plt.figure(figsize=(10, 6))
# train_df['Item_Outlet_Sales'].plot(kind='kde')  # Kernel Density Estimate curve
# submission_df['Item_Outlet_Sales'].plot(kind='kde')  # Kernel Density Estimate curve
plt.hist(train_df['Item_Outlet_Sales'], color='lightgreen', ec='black', bins=15)
plt.hist(submission_df['Item_Outlet_Sales'], color='red', ec='black', bins=15)
plt.title('Distribution Curve for Item Outlet Sales')
plt.xlabel('Item_Outlet Sales')
plt.ylabel('Density')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
# train_df['Item_MRP'].plot(kind='kde')  # Kernel Density Estimate curve
# test_df['Item_MRP'].plot(kind='kde')  # Kernel Density Estimate curve
plt.hist(train_df['Item_MRP'], color='lightgreen', ec='black', bins=15)
plt.hist(test_df['Item_MRP'], color='red', ec='black', bins=15)
plt.title('Distribution Curve for Item_MRP')
plt.xlabel('Item_MRP')
plt.ylabel('Density')
plt.grid(True)
plt.show()