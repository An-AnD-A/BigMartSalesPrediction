import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.helpers.config import train_data_path, test_data_path, output_base_path

# Data to be imputed

impute_data = {
    'OUT010': {
        'Outlet_Size' : 'Small',
    },
    'OUT017': {
        'Outlet_Size' : 'Small',
    },
    'OUT045': {
        'Outlet_Size' : 'Small',
    },
}

with open(output_base_path / 'outlet_mapping.json', 'r') as f:
    outlet_metadata = json.load(f)

for outlet_id, updates in impute_data.items():
    if outlet_id in outlet_metadata:
        outlet_metadata[outlet_id].update(updates)

# Save the updated outlet metadata
with open(output_base_path / 'outlet_mapping.json', 'w') as f:
    json.dump(outlet_metadata, f, indent=4)
    

if __name__ == "__main__":
    
    print(outlet_metadata)

