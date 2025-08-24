import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.helpers.config import train_data_path, test_data_path
from src.helpers.DataReader import filereader

from utils.DataInsights import (get_product_details, 
                                        get_outlet_details, 
                                        get_unique_item_list,
                                        get_item_outlet_mapping)


def main():

    # Read the train and test datasets
    train_df = filereader(train_data_path)
    test_df = filereader(test_data_path)

    train_df.info()
    test_df.info()

    # How many datapoints are available in the train and test datasets?
    print(f"Number of rows in train dataset: {train_df.shape[0]}")
    print(f"Number of rows in test dataset: {test_df.shape[0]}")

    # How many unique items are present in the dataset?
    item_count_train = get_unique_item_list(train_df)
    item_count_test = get_unique_item_list(test_df)
    print(f"Unique items in train dataset: {len(item_count_train)}")
    print(f"Unique items in test dataset: {len(item_count_test)}")

    # Are the items in test dataset also present in the train dataset?
    common_items = set(item_count_train).intersection(set(item_count_test))
    print(f"Common items between train and test datasets: {len(common_items)}")

    # Are the item and outlet combinations unique?
    item_outlet_train = get_item_outlet_mapping(train_df)
    item_outlet_test = get_item_outlet_mapping(test_df)

    print(f"Unique item-outlet combinations in train dataset: {len(item_outlet_train)}")
    print(f"Unique item-outlet combinations in test dataset: {len(item_outlet_test)}")

    # Are the item and outlet combinations in the test dataset also present in the train dataset?
    common_item_outlet = set(item_outlet_train.apply(tuple, axis=1)).intersection(set(item_outlet_test.apply(tuple, axis=1)))
    print(f"Common item-outlet combinations between train and test datasets: {len(common_item_outlet)}")

    # Can the same item have different item specs?
    product_details_train = get_product_details(train_df)
    product_details_test = get_product_details(test_df) 
    print(f"Unique product details in train dataset: {product_details_train.shape[0]}")
    print(f"Unique product details in test dataset: {product_details_test.shape[0]}")
    print(f'Unique products in train as per the product details df:{product_details_train["Item_Identifier"].nunique()}')
    print(f'Unique products in test as per the product details df:{product_details_test["Item_Identifier"].nunique()}')

    # Deep dive into product details and specs variations

    product_train_stats = product_details_train.groupby('Item_Identifier').agg({
        'Item_Weight': 'nunique',
        'Item_Fat_Content': 'nunique',
        'Item_Type': 'nunique',
        'Item_MRP': 'nunique'
    })

    print(product_train_stats)


    return


if __name__ == "__main__":

    # Run the main function
    main()