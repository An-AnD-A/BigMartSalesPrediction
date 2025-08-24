import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from utils.DataProcessing import item_mapper, map_item_fat_content
from src.helpers.config import train_data_path, test_data_path, output_base_path
from src.helpers.DataReader import filereader
from src.helpers.formatter import check_if_nan


def main():

    train_df = filereader(train_data_path)
    test_df = filereader(test_data_path)

    # Create initial item mapping
    metadata_dict = {}

    for index, row in train_df.iterrows():
        metadata_dict = item_mapper(
            item_identifier = row['Item_Identifier'], 
            item_weight=check_if_nan(row['Item_Weight']), 
            item_type=check_if_nan(row['Item_Type']), 
            item_fat_content=map_item_fat_content(check_if_nan(row['Item_Fat_Content'])), 
            item_dict=metadata_dict)
        
    print(f'Total items mapped: {len(metadata_dict)}')

    for index, row in test_df.iterrows():
        metadata_dict = item_mapper(
            item_identifier = row['Item_Identifier'], 
            item_weight=check_if_nan(row['Item_Weight']), 
            item_type=check_if_nan(row['Item_Type']), 
            item_fat_content=map_item_fat_content(check_if_nan(row['Item_Fat_Content'])), 
            item_dict=metadata_dict)
        
    print(f'Total items mapped: {len(metadata_dict)}')

    json_file = output_base_path / 'item_mapping.json'
    with open(json_file, 'w') as f:
        json.dump(metadata_dict, f, indent=4)

    return  

if __name__ == "__main__":
    main() 