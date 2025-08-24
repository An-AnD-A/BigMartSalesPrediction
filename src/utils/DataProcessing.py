import pandas as pd
import numpy as np
import json

from helpers.config import train_data_path, test_data_path, output_base_path
from helpers.DataReader import filereader


def item_mapper(item_identifier, 
                    item_weight, 
                    item_type, 
                    item_fat_content, 
                    item_dict):
    """
    Create a mapping for item characteristics.
    """

    if item_identifier not in item_dict.keys():
        print(f"Item {item_identifier} not found in item_dict, adding it.")
        # Initialize the item with its characteristics
        item_dict[item_identifier] = {
            'Item_Weight': item_weight,
            'Item_Type': item_type,
            'Item_Fat_Content': item_fat_content
        }
    else:
        print(f"Item {item_identifier} already exists in item_dict, checking for updates.")
        # Update the item characteristics if they are not already set
        if pd.isna(item_dict[item_identifier]['Item_Weight']) or item_dict[item_identifier]['Item_Weight'] in [np.nan, None, 0, 'NaN', 'nan', '']:
            item_dict[item_identifier]['Item_Weight'] = item_weight
            print(f"Item Weight for {item_identifier} is None, setting to {item_weight}")
        elif pd.isna(item_dict[item_identifier]['Item_Type']) or item_dict[item_identifier]['Item_Type'] in [np.nan, None, 0, 'NaN', 'nan', '']:
            item_dict[item_identifier]['Item_Type'] = item_type
            print(f"Item Type for {item_identifier} is None, setting to {item_type}")
        elif pd.isna(item_dict[item_identifier]['Item_Fat_Content']) or item_dict[item_identifier]['Item_Fat_Content'] in [np.nan, None, 0, 'NaN', 'nan', '']:
            item_dict[item_identifier]['Item_Fat_Content'] = item_fat_content
            print(f"Item Fat Content for {item_identifier} is None, setting to {item_fat_content}")

    return item_dict

def outlet_mapper(outlet_identifier, 
                  outlet_establishment_year, 
                  outlet_size,
                  outlet_location_type, 
                  outlet_type,
                  outlet_dict):
    
    """
    Create a mapping for outlet characteristics.
    """

    if outlet_identifier not in outlet_dict.keys():
        print(f"Outlet {outlet_identifier} not found in outlet_dict, adding it.")
        # Initialize the outlet with its characteristics
        outlet_dict[outlet_identifier] = {
            'Outlet_Establishment_Year': outlet_establishment_year,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_location_type,
            'Outlet_Type': outlet_type
        }
    else:
        print(f"Outlet {outlet_identifier} already exists in outlet_dict, checking for updates.")
        # Update the outlet characteristics if they are not already set
        if pd.isna(outlet_dict[outlet_identifier]['Outlet_Establishment_Year']) or outlet_dict[outlet_identifier]['Outlet_Establishment_Year'] in [np.nan, None, 0, 'NaN', 'nan', '']:
            outlet_dict[outlet_identifier]['Outlet_Establishment_Year'] = outlet_establishment_year
            print(f"Outlet Establishment Year for {outlet_identifier} is None, setting to {outlet_establishment_year}")
        if pd.isna(outlet_dict[outlet_identifier]['Outlet_Size']) or outlet_dict[outlet_identifier]['Outlet_Size'] in [np.nan, None, 0, 'NaN', 'nan', '']:
            outlet_dict[outlet_identifier]['Outlet_Size'] = outlet_size
            print(f"Outlet Size for {outlet_identifier} is None, setting to {outlet_size}")
        if pd.isna(outlet_dict[outlet_identifier]['Outlet_Location_Type']) or outlet_dict[outlet_identifier]['Outlet_Location_Type'] in [np.nan, None, 0, 'NaN', 'nan', '']:
            outlet_dict[outlet_identifier]['Outlet_Location_Type'] = outlet_location_type
            print(f"Outlet Location Type for {outlet_identifier} is None, setting to {outlet_location_type}")
        if pd.isna(outlet_dict[outlet_identifier]['Outlet_Type']) or outlet_dict[outlet_identifier]['Outlet_Type'] in [np.nan, None, 0, 'NaN', 'nan', '']:
            outlet_dict[outlet_identifier]['Outlet_Type'] = outlet_type
            print(f"Outlet Type for {outlet_identifier} is None, setting to {outlet_type}")

    return outlet_dict
    



    return item_dict

def create_initial_item_mapping(df=None):

    if not df:
        df = filereader(train_data_path)

    item_dict = {}
    for index, row in df.iterrows():
        item_mapper(row['Item_Identifier'], 
                    row['Item_Weight'], 
                    row['Item_Type'], 
                    row['Item_Fat_Content'], 
                    item_dict)
        
    print(f'Total items mapped: {len(item_dict)}')

    json_file = output_base_path / 'item_mapping.json'
    with open(json_file, 'w') as f:
        json.dump(item_dict, f, indent=4)

    return


def map_item_fat_content(value):
    if value in ['LF', 'low fat', 'Low Fat', ]:
        return 'Low Fat'
    elif value in ['reg', 'Regular']:
        return 'Regular'
    else:
        return 'Unknown'
    
def detect_outliers(df, feature):
    Q1  = df[feature].quantile(0.25)
    Q3  = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    return upper_limit, lower_limit    
    

if __name__ == "__main__":
    # Create item mapping for train dataset
    create_initial_item_mapping()