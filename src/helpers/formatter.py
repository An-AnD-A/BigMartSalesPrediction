import pandas as pd

def check_if_nan(value):
    """
    Check if the value is NaN or None.
    """
    if pd.isna(value):
        return 'nan'
    else:
        return value
    
