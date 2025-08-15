import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.helpers.config import train_data_path, test_data_path, output_base_path
from src.helpers.DataReader import filereader

def main():

    train_df = filereader(train_data_path)
    test_df = filereader(test_data_path)

    # Outlet characteristics overview
    train_outlet_df = train_df[['Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type','Outlet_Type']].drop_duplicates()
    train_outlet_agg = train_outlet_df.groupby('Outlet_Identifier').agg(
        Outlet_Establishment_Year_Unique_Count=('Outlet_Establishment_Year', 'nunique'),
        Outlet_Establishment_Year_Unique_Value=('Outlet_Establishment_Year', lambda x: list(set(x))),
        Outlet_Size_Unique_Count=('Outlet_Size', 'nunique'),
        Outlet_Size_Unique_Value=('Outlet_Size', lambda x: list(set(x))),
        Outlet_Location_Type_Unique_Count=('Outlet_Location_Type', 'nunique'),
        Outlet_Location_Type_Unique_Value=('Outlet_Location_Type', lambda x: list(set(x))),
        Outlet_Type_Unique_Count=('Outlet_Type', 'nunique'),
        Outlet_Type_Unique_Value=('Outlet_Type', lambda x: list(set(x))),
    )

    train_outlet_agg.to_csv(output_base_path / 'train_outlet_characteristics.csv')

    test_outlet_df = test_df[['Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type','Outlet_Type']].drop_duplicates()
    test_outlet_agg = test_outlet_df.groupby('Outlet_Identifier').agg(
        Outlet_Establishment_Year_Unique_Count=('Outlet_Establishment_Year', 'nunique'),
        Outlet_Establishment_Year_Unique_Value=('Outlet_Establishment_Year', lambda x: list(set(x))),
        Outlet_Size_Unique_Count=('Outlet_Size', 'nunique'),
        Outlet_Size_Unique_Value=('Outlet_Size', lambda x: list(set(x))),
        Outlet_Location_Type_Unique_Count=('Outlet_Location_Type', 'nunique'),
        Outlet_Location_Type_Unique_Value=('Outlet_Location_Type', lambda x: list(set(x))),
        Outlet_Type_Unique_Count=('Outlet_Type', 'nunique'),
        Outlet_Type_Unique_Value=('Outlet_Type', lambda x: list(set(x))),
    )
    test_outlet_agg.to_csv(output_base_path / 'test_outlet_characteristics.csv')

    return

if __name__ == "__main__":
    main()


