import pandas as pd

def filereader(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: DataFrame containing the data from the CSV file.

    Developer Note:
    ---------------
    - Expand the functionality to handle different file formats in the future.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None
    
