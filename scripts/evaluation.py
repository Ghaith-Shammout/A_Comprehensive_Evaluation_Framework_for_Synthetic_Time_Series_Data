import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance as wd

# Define the preprocessing pipeline
def population_fidelity_measure(real_file, synth_folder):
    """
    A complete evaluation measurmeents time series data.

    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to save the preprocessed CSV file.
    - columns (list): 
    """
    # Metric 1: 
    #metric(real_data, synth_folder, columns)

    # Metric 2: MSAS
    msas(real_file, synth_folder)

    # Metric 3: Average Wasserstien Distance
    awd(real_file, synth_folder)

    # Metric 4: Average Jehnsson-Shannon Distance
    #ajsd(real_data, synth_folder, columns)

    # Metric 5: Average Jehnsson-Shannon Distance
    #metric(real_data, synth_folder, columns)

    print(f"[+] Population Fidelity Completed")


def extract_sequences(df, date_column='date', seq_id='SID'):
    # Convert the 'date_column' to datetime format, specifying the exact format for conversion.
    # This ensures that the dates are in a standardized datetime object for further sorting.
    df[date_column] = pd.to_datetime(df[date_column], format='%d/%m/%Y %H:%M')
    
    # Initialize an empty list to store the sequences of values for each 'SID' group.
    sequences = []
    
    # Group the DataFrame by sequence id (seq_id), iterating over each unique group.
    # 'SID' could represent an identifier for different sequences, e.g., a user or entity.
    for _, group in df.groupby(seq_id):
        # Sort each group by the 'date_column' so that we can process the data in chronological order.
        group_sorted = group.sort_values(by=date_column)
        
        # Drop the 'date_column' as it's no longer needed after sorting, we only need the sequence of values.
        group_sorted = group_sorted.drop(columns=[date_column])
        
        # Append the values of the sorted group (as a numpy array) to the 'sequences' list.
        # This stores each group as a sequence of rows.
        sequences.append(group_sorted.values)
    
    # Return the list of sequences where each entry corresponds to a sorted sequence for a particular 'SID'.
    return sequences

def calculate_sequence_statistics(sequences):
    # Initialize a dictionary to store the statistics for each sequence.
    # Keys: 'length', 'mean', 'median', 'std', 'inter_row_diff' correspond to different statistics.
    stats = {'length': [], 'mean': [], 'median': [], 'std': [], 'inter_row_diff': []}
    
    # Loop through each sequence in the list of sequences.
    for seq in sequences:
        # Calculate and store the length (number of rows) of the sequence.
        stats['length'].append(len(seq))
        
        # Calculate and store the mean of each column (axis=0 means column-wise mean).
        stats['mean'].append(np.mean(seq, axis=0))
        
        # Calculate and store the median of each column (axis=0 means column-wise median).
        stats['median'].append(np.median(seq, axis=0))
        
        # Calculate and store the standard deviation of each column (axis=0 means column-wise std).
        stats['std'].append(np.std(seq, axis=0))
        
        # Initialize a list to store the "inter-row differences" for each column.
        inter_row_diffs = []
        
        # For each column in the sequence, calculate the difference between values
        # that are 24 rows apart (lag of 24).
        for col in range(seq.shape[1]):  # Iterate over all columns
            col_vals = seq[:, col]  # Extract values of the current column
            
            # If the sequence has more than 24 rows, calculate the average absolute difference
            # between each value and the value 24 rows later (lag of 24).
            # If the sequence has less than 24 rows, set the difference to NaN.
            diff = np.mean([abs(col_vals[i] - col_vals[i + 24]) for i in range(len(col_vals) - 24)]) if len(col_vals) > 24 else np.nan
            inter_row_diffs.append(diff)
        
        # Store the list of inter-row differences for the current sequence.
        stats['inter_row_diff'].append(inter_row_diffs)
    
    # Return the dictionary containing all the statistics for each sequence.
    return stats

def ks_test_between_columns(real_column_stats, synth_column_stats):
    # Initialize a dictionary to store the KS test results for each statistic (mean, median, std, inter-row diff).
    # The values will be lists of tuples (KS statistic, p-value) for each statistic.
    ks_results = {'mean': [], 'median': [], 'std': [], 'inter_row_diff': []}
    
    # Loop through each statistic name in the keys of the ks_results dictionary.
    for stat_name in ks_results.keys():
        # Retrieve the corresponding statistics for the real and synthetic columns.
        real_stat = real_column_stats[stat_name]  # Statistics from the real column
        synth_stat = synth_column_stats[stat_name]  # Statistics from the synthetic column
        
        # Perform the Kolmogorov-Smirnov (KS) two-sample test between the real and synthetic statistics.
        # ks_2samp returns the KS statistic and the p-value.
        ks_stat, p_value = ks_2samp(real_stat, synth_stat)
        
        # Store the results (KS statistic and p-value) for the current statistic name.
        ks_results[stat_name].append((ks_stat, p_value))
    
    # Return the dictionary containing the KS test results for each statistic.
    return ks_results

def calculate_msas(real_df, synth_df):
    # Extract sequences from both the real and synthetic dataframes.
    # These sequences will be used to compute statistics (e.g., mean, median, etc.).
    real_sequences = extract_sequences(real_df)
    synth_sequences = extract_sequences(synth_df)
    
    # Calculate statistics (mean, median, std, inter-row diff) for the real and synthetic sequences.
    real_stats = calculate_sequence_statistics(real_sequences)
    synth_stats = calculate_sequence_statistics(synth_sequences)
    
    # Get the column names excluding 'SID' and 'date' (which are not part of the data to be analyzed).
    column_names = real_df.columns[:-2]  # Excludes 'SID' and 'date' columns
    msas_results = {}  # Dictionary to store MSAS results for each column
    
    # Loop through each column (excluding 'SID' and 'date') to perform the KS test and calculate MSAS results.
    for col_idx, col_name in enumerate(column_names):
        # Prepare the statistics for the real column at the current index.
        # Extract statistics (mean, median, std, inter-row diff) for the current column (col_idx).
        real_col_stats = {
            'mean': [real_stats['mean'][i][col_idx] for i in range(len(real_stats['mean']))],
            'median': [real_stats['median'][i][col_idx] for i in range(len(real_stats['median']))],
            'std': [real_stats['std'][i][col_idx] for i in range(len(real_stats['std']))],
            'inter_row_diff': [real_stats['inter_row_diff'][i][col_idx] for i in range(len(real_stats['inter_row_diff']))]
        }

        # Prepare the statistics for the synthetic column at the current index.
        synth_col_stats = {
            'mean': [synth_stats['mean'][i][col_idx] for i in range(len(synth_stats['mean']))],
            'median': [synth_stats['median'][i][col_idx] for i in range(len(synth_stats['median']))],
            'std': [synth_stats['std'][i][col_idx] for i in range(len(synth_stats['std']))],
            'inter_row_diff': [synth_stats['inter_row_diff'][i][col_idx] for i in range(len(synth_stats['inter_row_diff']))]
        }

        # Perform the Kolmogorov-Smirnov test between the real and synthetic statistics for the current column.
        ks_results = ks_test_between_columns(real_col_stats, synth_col_stats)
        
        # Calculate the MSAS (Mean Statistical Agreement Score) for each statistic.
        # This is done by averaging the KS statistics (the first value in each tuple from the KS test results).
        msas_results[col_name] = {
            stat: np.mean([result[0] for result in ks_results[stat]]) for stat in ks_results
        }
        
    # Return the MSAS results for all columns.
    return msas_results

def plot_msas_grouped(msas_results, synth_name, save_path='./'):
    # Get the list of column names from the MSAS results (keys of the dictionary).
    columns = list(msas_results.keys())
    
    # Define the statistics to be plotted (mean, median, std, and inter_row_diff).
    statistics = ['mean', 'median', 'std', 'inter_row_diff']
    
    # Create a new figure and axis object for the plot with a specified size (12x8 inches).
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Loop through each statistic type (mean, median, std, inter_row_diff).
    for i, stat in enumerate(statistics):
        # Extract the KS statistic values for the current statistic (mean, median, etc.) for all columns.
        ks_values = [msas_results[col][stat] for col in columns]
        
        # Create a bar plot for the current statistic (mean, median, std, or inter-row difference).
        # Each statistic will have its bars slightly offset to avoid overlapping.
        # The offset is determined by `i * 0.2` to ensure bars for different statistics are grouped but not overlapping.
        ax.bar(np.arange(len(columns)) + i * 0.2, ks_values, 0.2, label=stat)
    
    # Set the x-axis label (representing the column names).
    ax.set_xlabel('Columns')
    
    # Set the y-axis label (representing the KS Statistic value).
    ax.set_ylabel('KS Statistic')
    
    # Set the title of the plot, which includes the name of the synthetic dataset (synth_name).
    ax.set_title(f'MSAS Results: KS Statistic for {synth_name}')
    
    # Set the x-ticks to correspond to the positions of the columns on the x-axis.
    # The x-tick positions are adjusted to ensure that the bars for each statistic are grouped correctly.
    ax.set_xticks(np.arange(len(columns)) + (len(statistics) - 1) * 0.1)
    
    # Set the x-tick labels to the actual column names, rotating them by 45 degrees for better readability.
    ax.set_xticklabels(columns, rotation=45, ha="right")
    
    # Add a legend to the plot to indicate what each color bar represents (mean, median, std, inter-row_diff).
    ax.legend(title='Statistic Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust the layout to prevent clipping of axis labels and the legend.
    plt.tight_layout()
    
     # If a save_path is provided, ensure the directory exists and then save the plot as a file.
    if save_path:
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the figure to the given path
        plt.savefig(save_path, bbox_inches='tight')  
        print(f"Plot saved as {save_path}")
    
    # Display the plot.
    #plt.show()

def write_msas_results_to_file(msas_results, synth_file, output_file):
    # Open the output file in write mode
    with open(output_file, 'w') as file:
        # Write the MSAS results for the current synthetic dataset
        file.write(f"MSAS results for synthetic dataset: {synth_file}\n")
        
        # Loop through the results for each column in the MSAS output
        for column, ks_stats in msas_results.items():
            file.write(f"  - {column}:\n")  # Write the column name
            
            # Loop through each statistic type (mean, median, std, inter_row_diff)
            # and write the KS statistic value for that statistic
            for stat, ks_stat in ks_stats.items():
                file.write(f"    {stat}: KS Statistic = {ks_stat:.4f}\n")
        
        # Write a separator to distinguish between results of different datasets
        file.write("\n" + "-"*50 + "\n")
    print(f"[+] MSAS results saved to {output_file}")

def run_msas_for_synthetic_datasets(real_df, synth_dir):
    # Get a list of all CSV files in the given synthetic data directory
    # The list comprehension filters only files that end with '.csv'
    synthetic_files = [f for f in os.listdir(synth_dir) if f.endswith('.csv')]
    
    # Loop through each synthetic dataset file in the directory
    for synth_file in synthetic_files:
        # Read the synthetic dataset CSV file into a DataFrame
        synth_df = pd.read_csv(os.path.join(synth_dir, synth_file))
        
        # Run the MSAS (Mean Statistical Agreement Score) analysis on the real and synthetic data
        msas_results = calculate_msas(real_df, synth_df)
        
        # Plot MSAS results for the current synthetic dataset
        plot_msas_grouped(msas_results, synth_file, f'./outputs/Plots/{synth_file}.png')  # Plot the KS statistics for each column
        
        
        # Example of usage
        output_file = './outputs/Evaluation/msas_results.txt'  # Specify the output file path
        
        # If a save_path is provided, ensure the directory exists and then save the plot as a file.
        if output_file:
            # Ensure the directory exists
            save_dir = os.path.dirname(output_file)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
        write_msas_results_to_file(msas_results, synth_file, output_file)

def msas(real_file, synth_folder):
    # Load the real dataset
    real_data = pd.read_csv(real_file)

    # Columns 
    columns = [
        'Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)',
        'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'SID', 'date'
    ]
    
    real_df = real_data[columns]

    # Define the directory containing synthetic datasets
    synth_dir = synth_folder

    # Run MSAS for all synthetic datasets in the directory
    run_msas_for_synthetic_datasets(real_df, synth_dir)
    print(f"[+] MSAS Completed")


def awd(real_file, synth_data):
    """
    Calculate the Average Wasserstein Distance between Real data and Synthetic datasets
    """
    print("[+] Calculating Average Wasserstein Distance (AWD)")

    # Read the real dataset
    real = pd.read_csv(real_file)

    # List of columns to compare
    columns = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
               'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)',
               'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor']


    # Initialize a dictionary to store lists of WD values for each column
    wd_values_dict = {column: [] for column in columns}

    # Path to the directory containing synthetic data files
    synth_data_dir = synth_data

    # Get a list of synthetic data files (assuming they are CSV files)
    synth_files = [f for f in os.listdir(synth_data_dir) if f.endswith('.csv')]

    # Loop through each synthetic dataset
    for synth_file in synth_files:
        # Read the synthetic data
        synth = pd.read_csv(os.path.join(synth_data_dir, synth_file))

        # Loop through each column and calculate the Wasserstein distance
        for column in columns:
            wd_value = wd(real[column], synth[column])
            wd_values_dict[column].append(wd_value)

    # Plotting the Wasserstein distance for each column over different synthetic datasets
    plt.figure(figsize=(14, 8))

    # Color palette for distinct lines
    colors = plt.cm.viridis(np.linspace(0, 1, len(columns)))

    # Plot each column's WD values over synthetic datasets
    for idx, (column, wd_values) in enumerate(wd_values_dict.items()):
        plt.plot(range(len(wd_values)), wd_values, marker='o', label=column, color=colors[idx], markersize=6, linewidth=2)

    # Set plot labels and title
    plt.xlabel('Index of Synthetic Dataset', fontsize=12)
    plt.ylabel('Wasserstein Distance (WD)', fontsize=12)
    plt.title('Evolution of Wasserstein Distance between Real and Synthetic Data for Different Columns', fontsize=14)

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set the x-ticks to be the indices of synthetic datasets
    plt.xticks(range(len(synth_files)), synth_files, rotation=45, ha='right', fontsize=10)

    # Display legend with more clarity
    plt.legend(title="Columns", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Tight layout to prevent clipping of labels
    plt.tight_layout()

    # Show the plot
    #plt.show()

    output_dir = './outputs/Plots/'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    plt.savefig(os.path.join(output_dir, 'awd.png'), bbox_inches='tight')

    print(f"[+] Average Wasserstein Distance Completed")
    

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run evaluation metrics for synthetic data.")
    parser.add_argument("--real_data", required=True, help="Path to the real data CSV file.")
    parser.add_argument("--synth_folder", required=True, help="Path to the folder containing synthetic data CSV files.")
    args = parser.parse_args()

    # Run the evaluation
    population_fidelity_measure(args.real_data, args.synth_folder)