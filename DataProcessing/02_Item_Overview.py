from src.helpers.config import train_data_path, test_data_path, output_base_path
from src.helpers.DataReader import filereader

"""
ObaervaTions:
1. Item Fat Content has just 2 unique values either 'low fat' or 'Regular'.
2. Item weight is unique across 

"""
def main():

    # Read the train and test datasets
    train_df = filereader(train_data_path)
    test_df = filereader(test_data_path)

    # Item and their characteristics - Does the item characteristics vary across different outlets?

    # We know that the MRP and visibility changes across the outlets. What about the other characteristics?

    train_item_df = train_df[['Item_Identifier', 'Item_Weight', 'Item_Type', 'Item_Fat_Content']].drop_duplicates()
    train_item_agg = train_item_df.groupby('Item_Identifier').agg(
        Item_Weight_Unique_Count=('Item_Weight', 'nunique'),
        Item_Weight_Unique_Value=('Item_Weight', lambda x: list(set(x))),
        Item_FatContent_Unique_Count=('Item_Fat_Content', 'nunique'),
        Item_FatContent_Unique_Value=('Item_Fat_Content', lambda x: list(set(x))),
        Item_Type_Unique_Count=('Item_Type', 'nunique'),
        Item_Type_Unique_Value=('Item_Type', lambda x: list(set(x))),
    )

    train_item_agg.to_csv(output_base_path / 'train_item_characteristics.csv')

    test_item_df = test_df[['Item_Identifier', 'Item_Weight', 'Item_Type', 'Item_Fat_Content']].drop_duplicates()
    test_item_agg = test_item_df.groupby('Item_Identifier').agg(
        Item_Weight_Unique_Count=('Item_Weight', 'nunique'),
        Item_Weight_Unique_Value=('Item_Weight', lambda x: list(set(x))),
        Item_FatContent_Unique_Count=('Item_Fat_Content', 'nunique'),
        Item_FatContent_Unique_Value=('Item_Fat_Content', lambda x: list(set(x))),
        Item_Type_Unique_Count=('Item_Type', 'nunique'),
        Item_Type_Unique_Value=('Item_Type', lambda x: list(set(x))),
    )

    test_item_agg.to_csv(output_base_path / 'test_item_characteristics.csv')

    # Item 

    return 


if __name__ == "__main__":
    main()
    
