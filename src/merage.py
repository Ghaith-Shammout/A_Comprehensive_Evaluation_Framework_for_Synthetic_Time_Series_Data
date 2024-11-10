import pandas as pd
import glob
import os

def merge_csv_files(directory, output_file):
    # Create a pattern to match CSV files
    csv_pattern = os.path.join(directory, '*.csv')
    
    # Read all CSV files into a list of DataFrames
    csv_files = glob.glob(csv_pattern)
    dataframes = [pd.read_csv(file) for file in csv_files]
    
    # Merge all DataFrames into one
    merged_data = pd.concat(dataframes, ignore_index=True)
    
    # Save the merged DataFrame to a new CSV file
    merged_data.to_csv(output_file, index=False)
    print(f"Merged {len(csv_files)} files into {output_file}")

