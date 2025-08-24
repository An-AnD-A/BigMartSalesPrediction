import pandas as pd 

def get_unique_item_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the list of unique items in the dataset.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing sales data.
    
    Returns:
    pd.DataFrame: DataFrame containing unique items.
    """
    
    item_list = df['Item_Identifier'].unique()
    
    return item_list

def get_product_details(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get details of products in the dataset.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing sales data.
    
    Returns:
    pd.DataFrame: DataFrame containing product details.
    """
    
    product_details = df[['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Type', 'Item_MRP']].drop_duplicates()
    
    return product_details

def get_outlet_details(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get details of outlets in the dataset.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing sales data.
    
    Returns:
    pd.DataFrame: DataFrame containing outlet details.
    """
    
    outlet_details = df[['Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']].drop_duplicates()
    
    return outlet_details

def get_item_outlet_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get unique combinations of items and outlets.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing sales data.
    
    Returns:
    pd.DataFrame: DataFrame containing unique item-outlet combinations.
    """
    
    item_outlet_mapping = df[['Item_Identifier', 'Outlet_Identifier']].drop_duplicates()
    
    return item_outlet_mapping