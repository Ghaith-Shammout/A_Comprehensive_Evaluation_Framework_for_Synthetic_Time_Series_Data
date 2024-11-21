import pandas as pd
import glob
import os

import pandas as pd

def enforce_date_format(file_path, output_path=file_path):
    """
    Ensures all values in the 'date' column follow the format 'dd/MM/yyyy HH:mm'.
    :param file_path: Path to the input CSV file.
    :param output_path: Path to save the updated CSV file.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Ensure the 'date' column exists
    if 'date' not in df.columns:
        raise ValueError("The 'date' column is not found in the dataset.")

    # Parse the 'date' column and reformat it
    try:
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')  # Parse the dates
    except ValueError as e:
        raise ValueError(f"Error parsing dates: {e}. Please check your data.")

    # Format the 'date' column to the desired format
    df['date'] = df['date'].dt.strftime('%d/%m/%Y %H:%M')

    # Save the updated DataFrame to a new CSV
    df.to_csv(output_path, index=False)
    print(f"Date format standardized and saved to '{output_path}'.")


def sliding_window(df, window_size=50, step_size=1):
    """
    Generates sliding windows from the DataFrame.

    This function splits a DataFrame into smaller overlapping DataFrames (windows) 
    based on the specified window size and step size.

    :param df: The input DataFrame to split.
    :param window_size: The size of each sliding window (number of rows per window).
    :param step_size: The step size for sliding the window. This determines 
                       how many rows to move the window after each iteration.
    :return: A DataFrame containing all the sliding windows.
    """
    windows = []
    
    # Loop through the DataFrame, creating windows starting at different positions
    for start in range(0, len(df) - window_size + 1, step_size):
        # Extract a window of data and reset the index within each window
        window = df.iloc[start:start + window_size].reset_index(drop=True)
        windows.append(window)
    
    # Concatenate all the windows into a single DataFrame
    return pd.concat(windows, ignore_index=True)

def process_csv_files_from_directory(input_dir, output_file, window_size=50, step_size=1):
    """
    Processes all CSV files in a directory, applies the sliding window technique 
    to each file, and saves the result to an output CSV.

    This function reads each CSV file in the specified directory, applies the sliding 
    window transformation to each, and then combines the resulting windows into a 
    single DataFrame, which is saved as a new CSV file.

    :param input_dir: Directory containing the input CSV files to process.
    :param output_file: The path of the output CSV file to save the results.
    :param window_size: The size of each sliding window (number of rows per window).
    :param step_size: The step size for the sliding window. This determines 
                       how much to move the window after each iteration.
    """
    all_windows = []
    
    # Get list of all CSV files in the input directory
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    # Process each CSV file in the directory
    for file in input_files:
        file_path = os.path.join(input_dir, file)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Apply the sliding window technique to the DataFrame
        windows_df = sliding_window(df, window_size, step_size)
        
        # Append the result (sliding windows of the current file) to the list
        all_windows.append(windows_df)

    # Combine all the windowed DataFrames into a single DataFrame
    result = pd.concat(all_windows, ignore_index=True)

    # Save the resulting DataFrame with all sliding windows to the output CSV
    result.to_csv(output_file, index=False)
    
    # Print a message indicating the processing is complete
    print(f"Sliding windows processed and saved to {output_file}")

