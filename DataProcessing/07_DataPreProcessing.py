import pandas as pd
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.helpers.config import train_data_path, test_data_path, output_base_path, item_metadata_path, outlet_metadata_path
from src.helpers.DataReader import filereader

def process_train_data():
    # Read the train dataset
    train_df = filereader(train_data_path)

    item_metadata = json.load(open(item_metadata_path, 'r'))
    outlet_metadata = json.load(open(outlet_metadata_path, 'r'))

    # Process the train dataset
    for idx, row in train_df.iterrows():

        specific_item_metadata = item_metadata[row['Item_Identifier']]
        specific_outlet_metadata = outlet_metadata[row['Outlet_Identifier']]

        ## Item Metadata Imputation
        if specific_item_metadata['Item_Weight'] != row['Item_Weight']:
            train_df.loc[idx, 'Item_Weight'] = specific_item_metadata['Item_Weight']
            print(f"Item Weight for {row['Item_Identifier']} is None, setting to {specific_item_metadata['Item_Weight']}")

        if specific_item_metadata['Item_Type'] != row['Item_Type']:
            train_df.loc[idx, 'Item_Type'] = specific_item_metadata['Item_Type']
            print(f"Item Type for {row['Item_Identifier']} is None, setting to {specific_item_metadata['Item_Type']}")

        if specific_item_metadata['Item_Fat_Content'] != row['Item_Fat_Content']:
            train_df.loc[idx, 'Item_Fat_Content'] = specific_item_metadata['Item_Fat_Content']
            print(f"Item Fat Content for {row['Item_Identifier']} is None, setting to {specific_item_metadata['Item_Fat_Content']}")

        ## Outlet Metadata Imputation
        if specific_outlet_metadata['Outlet_Size'] != row['Outlet_Size']:
            train_df.loc[idx, 'Outlet_Size'] = specific_outlet_metadata['Outlet_Size']
            print(f"Outlet Size for {row['Outlet_Identifier']} is None, setting to {specific_outlet_metadata['Outlet_Size']}")  

        if specific_outlet_metadata['Outlet_Location_Type'] != row['Outlet_Location_Type']:
            train_df.loc[idx, 'Outlet_Location_Type'] = specific_outlet_metadata['Outlet_Location_Type']
            print(f"Outlet Location Type for {row['Outlet_Identifier']} is None, setting to {specific_outlet_metadata['Outlet_Location_Type']}")

        if specific_outlet_metadata['Outlet_Type'] != row['Outlet_Type']:
            train_df.loc[idx, 'Outlet_Type'] = specific_outlet_metadata['Outlet_Type']
            print(f"Outlet Type for {row['Outlet_Identifier']} is None, setting to {specific_outlet_metadata['Outlet_Type']}")

    train_df.info()

    # Save the processed train dataset
    train_df.to_csv(output_base_path / 'processed_train_data.csv', index=False)

    return


if __name__ == "__main__":
    # Process the train data
    process_train_data()

    # Process the test data
    # Note: The test data processing can be similar to the train data processing.
    # You can implement a similar function for test data if needed.



        
        

