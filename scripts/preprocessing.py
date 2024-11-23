import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Define the preprocessing pipeline
def preprocess_data(input_file, output_file, unwanted_columns, date_column, categorical_columns, window_size, step_size, normalization_method='minmax'):
    """
    A complete preprocessing pipeline for time series data.

    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to save the preprocessed CSV file.
    - unwanted_columns (list): List of columns to remove.
    - date_column (str): Name of the column containing date values.
    - categorical_columns (list): List of categorical columns to encode.
    - window_size (int): Size of the sliding window.
    - step_size (int): Step size for the sliding window.
    - normalization_method (str, optional): Method for normalizing numerical columns ('minmax' or 'zscore').
    """
    
    # Step 1: Remove unwanted columns
    remove_unwanted_columns(input_file, 'step1_removed_columns.csv', unwanted_columns)

    # Step 2: Enforce date format
    enforce_date_format('step1_removed_columns.csv', date_column, 'step2_date_formatted.csv')

    # Step 3: Normalize numerical data
    normalize_data('step2_date_formatted.csv', 'step3_normalized.csv', method=normalization_method)

    # Step 4: Label encode categorical columns
    df = pd.read_csv('step3_normalized.csv')
    df, label_encoders = label_encode_and_save(df, categorical_columns, 'step4_encoded.csv')

    # Step 5: Apply sliding window
    windowed_df = sliding_window(df, window_size, step_size, output_file)
    windowed_df.to_csv(output_file, index=False)
    print(f"[+] Final preprocessed data saved to {output_file}.")


def remove_unwanted_columns(input_file, output_file, columns_to_remove):
    """
    Removes specified columns from a CSV file and saves the updated DataFrame to a new CSV file.

    Parameters:
    input_file (str): The path to the input CSV file from which unwanted columns will be removed.
    output_file (str): The path where the resulting CSV file will be saved.
    columns_to_remove (list of str): List of columns to be removed from the DataFrame.
    
    Returns:
    None
    """
    
    # Read the input CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)
    
    # Drop the columns specified in the columns_to_remove list
    df.drop(columns=columns_to_remove, inplace=True)
    
    # Save the updated DataFrame to the output CSV file (without the index)
    df.to_csv(output_file, index=False)
    
    # Print a message indicating which columns were removed and where the output was saved
    print(f"[+] Columns {columns_to_remove} removed. Output saved to {output_file}.")


def enforce_date_format(file_path, date_column, output_path=None, date_format="%d/%m/%Y %H:%M"):
    """
    Ensures all values in the specified 'date_column' follow the specified date format.

    Parameters:
    file_path (str): Path to the input CSV file.
    date_column (str): Name of the column containing dates that need to be reformatted.
    output_path (str, optional): Path to save the updated CSV file. If None, the changes will not be saved.
    date_format (str, optional): The desired date format. Default is "%d/%m/%Y %H:%M" (e.g., "21/11/2024 14:30").
    
    Returns:
    None
    """
    
    # Load the dataset from the provided file path into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Ensure the provided 'date_column' exists in the dataset
    if date_column not in df.columns:
        raise ValueError(f"The {date_column} column is not found in the dataset.")

    # Try to parse the 'date_column' values into datetime objects using the provided date format
    try:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)  # Parse the dates
    except ValueError as e:
        # Raise an error if the date format doesn't match the specified format
        raise ValueError(f"[-] Error parsing dates: {e}. Please check your data.")

    # Reformat the 'date_column' to the desired date format
    df[date_column] = df[date_column].dt.strftime(date_format)

    # If output_path is provided, save the updated DataFrame to a new CSV file
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[+] Date format standardized and saved to '{output_path}'.")
    else:
        print("[-] Date format standardized. No output file path provided, so data was not saved.")

def normalize_data(input_file, output_file, method='minmax'):
    """
    Normalizes all numerical columns in a dataset.
    :param input_file: Path to the input CSV file.
    :param output_file: Path to save the normalized CSV file.
    :param method: Normalization method ('minmax' or 'zscore').
    """
    # Load the dataset
    df = pd.read_csv(input_file)

    # Identify numerical columns
    numerical_cols = [
        'Usage_kWh', 
        'Lagging_Current_Reactive.Power_kVarh', 
        'Leading_Current_Reactive_Power_kVarh',
        'CO2(tCO2)', 
        'Lagging_Current_Power_Factor', 
        'Leading_Current_Power_Factor'
    ]

    # Select the normalization method
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError("[-] Unsupported normalization method. Use 'minmax' or 'zscore'.")

    # Normalize numerical columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Save the normalized dataset
    df.to_csv(output_file, index=False)
    print(f"[+] Normalized data saved to '{output_file}'.")


def label_encode_and_save(df, categorical_columns, output_file):
    """
    Function to label-encode categorical columns in a DataFrame and save the updated DataFrame to a file.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        categorical_columns (list): List of column names to label-encode.
        output_file (str): Path to save the updated DataFrame as a CSV file.
    
    Returns:
        pd.DataFrame: Updated DataFrame with encoded categorical columns.
    """
    # Create a dictionary to store mappings for each column
    label_encoders = {}
    
    for column in categorical_columns:
        # Initialize LabelEncoder
        label_encoder = LabelEncoder()
        
        # Encode the column
        df[column] = label_encoder.fit_transform(df[column])
        
        # Store the LabelEncoder for this column
        label_encoders[column] = label_encoder

        # Optionally, display mapping for reference
        print(f"[+] Encoding mapping for '{column}': {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Save the updated DataFrame to a file
    df.to_csv(output_file, index=False)
    print(f"[+] Updated DataFrame saved to {output_file}")
    
    return df, label_encoders


import pandas as pd

def sliding_window(data, window_size, step_size, output_file):
    """
    Applies the sliding window technique to a dataset.
    
    :param data: Pandas DataFrame containing the time series data.
    :param window_size: Size of each window (number of data points).
    :param step_size: Step size to slide the window.
    :param output_file: Path to save the sliding window data.
    :return: DataFrame containing the sliced windows with unique SIDs.
    """
    if len(data) < window_size:
        raise ValueError("The window size is larger than the dataset. Adjust the window size.")
    
    # Initialize a list to store windowed data
    windowed_data = []
    sid = 1  # Start the sequence identifier from 1

    # Iterate through the dataset with the specified window and step sizes
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = data.iloc[start:end].copy()  # Extract the window
        window['SID'] = sid  # Assign a unique SID to the window
        windowed_data.append(window)
        sid += 1  # Increment the sequence identifier

    print(f"[+] Sliding window successfully performed, with number of windows {len(windowed_data)}")

    # Combine all windows into a single DataFrame
    slided_data = pd.concat(windowed_data, axis=0)

    # Save the sliding window dataset
    slided_data.to_csv(output_file, index=False)
    print(f"[+] Sliding window data saved to '{output_file}'.")
    
    return slided_data


