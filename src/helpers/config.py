from pathlib import Path

def get_root_directory(marker_files = ('pyproject.toml', '.gitignore')) -> Path:
    """
    Returns the root directory of the project.
    """
    current_file = Path(__file__).resolve()

    for parent in [current_file] + list(current_file.parents):
        if all((parent / marker).exists() for marker in marker_files):
            return parent
        
    raise FileNotFoundError(f"Root directory not found. Marker files {marker_files} not found in any parent directories.")

data_root = Path.joinpath(get_root_directory(),'Data')

train_data_path = data_root / 'train.csv'
test_data_path = data_root / 'test.csv'

output_base_path = data_root / 'output'
    
if __name__ == "__main__":
    # Example usage
    try:
        root_dir = get_root_directory()
        print(f"Root directory: {root_dir}")
    except FileNotFoundError as e:
        print(e)

    print(f"Data directory: {data_root}")

    print(f"Train data path: {train_data_path}")
    print(f"Test data path: {test_data_path}")