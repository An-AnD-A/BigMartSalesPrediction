import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.helpers.config import train_data_path, test_data_path, output_base_path
from src.helpers.DataReader import filereader
from utils.viz import plot_frequency_distribution, plot_scatter_plot
from utils.DataProcessing import outlet_mapper
from src.helpers.formatter import check_if_nan


def main():

    train_df = filereader(train_data_path)
    test_df = filereader(test_data_path)

    # Plot the total item count and sales output from each outlet.

    outlet_train_df = train_df.groupby('Outlet_Identifier').agg(
        Total_Items_Sold=('Item_Identifier', 'count'),
        Total_Sales_Output=('Item_Outlet_Sales', 'sum')
    ).reset_index()

    print(outlet_train_df)

    plot_scatter_plot(
        outlet_train_df,
        x_column='Total_Items_Sold',
        y_column='Total_Sales_Output',
        title='Total Items Sold vs Total Sales Output in Train Dataset',
        label_column='Outlet_Identifier',
        )
    
    # Doing the same analysis without the grocery shop and supermarket type 3 outlets
    outlet_train_df_filtered = outlet_train_df[
        ~outlet_train_df['Outlet_Identifier'].isin(['OUT010', 'OUT019','OUT027'])
    ]

    plot_scatter_plot(
        outlet_train_df_filtered,
        x_column='Total_Items_Sold',
        y_column='Total_Sales_Output',
        title='Total Items Sold vs Total Sales Output in Train Dataset (Filtered)',
        label_column='Outlet_Identifier',
    )
    
    # Create initial item mapping
    metadata_dict = {}

    for index, row in train_df.iterrows():
        metadata_dict = outlet_mapper(
            outlet_identifier=row['Outlet_Identifier'], 
            outlet_establishment_year=check_if_nan(row['Outlet_Establishment_Year']), 
            outlet_size=check_if_nan(row['Outlet_Size']), 
            outlet_location_type=check_if_nan(row['Outlet_Location_Type']), 
            outlet_type=check_if_nan(row['Outlet_Type']),
            outlet_dict=metadata_dict
        )
        
    print(f'Total items mapped: {len(metadata_dict)}')

    json_file = output_base_path / 'outlet_mapping.json'
    with open(json_file, 'w') as f:
        json.dump(metadata_dict, f, indent=4)

    return 


if __name__ == "__main__":
    main()
