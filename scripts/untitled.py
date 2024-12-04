import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def compute_inter_row_dependency(sequence, lag=1):
    """
    Computes the average difference between a value in row n and the value `lag` steps after it in a sequence.
    """
    return np.mean(np.abs(sequence[:-lag] - sequence[lag:]))

def compute_statistics(df, sid_col='SID', exclude_cols=['date']):
    """
    Computes statistics for each sequence grouped by the sequence ID.
    """
    stats = []
    columns = [col for col in df.columns if col not in exclude_cols + [sid_col]]
    for sid, group in df.groupby(sid_col):
        seq_stats = {'SID': sid}
        seq_stats['length'] = len(group)
        for col in columns:
            data = group[col].values
            seq_stats[f'{col}_mean'] = np.mean(data)
            seq_stats[f'{col}_median'] = np.median(data)
            seq_stats[f'{col}_std'] = np.std(data)
            seq_stats[f'{col}_inter_row_dep'] = compute_inter_row_dependency(data, lag=1)
        stats.append(seq_stats)
    return pd.DataFrame(stats)

def compute_msas(real_stats, synthetic_stats):
    """
    Computes the MSAS score by averaging the Kolmogorov-Smirnov test results for all columns.
    """
    scores = []
    for column in real_stats.columns:
        if column == 'SID':  # Skip the sequence identifier column
            continue
        real_values = real_stats[column].dropna()
        synthetic_values = synthetic_stats[column].dropna()
        ks_stat, _ = ks_2samp(real_values, synthetic_values)
        scores.append(1 - ks_stat)  # Convert KS statistic to a similarity score
    return np.mean(scores)

def extract_epoch_number(filename):
    """
    Extracts the epoch number from the filename. Assumes the filename contains the word 'epoch' followed by a number.
    Example: synthetic_data_epoch1.csv -> epoch1
    """
    base_name = os.path.splitext(filename)[0]
    epoch_part = [part for part in base_name.split('_') if 'epoch' in part]
    return epoch_part[0] if epoch_part else base_name

# Main script
def msas_pipeline(real_dataset_path, synthetic_dataset_dir, output_csv, sid_col='SID', exclude_cols=['date']):
    """
    Executes the MSAS algorithm for the real and synthetic datasets, writing results to a CSV file.
    """
    # Load the real dataset
    real_df = pd.read_csv(real_dataset_path)

    # Compute statistics for the real dataset
    real_stats = compute_statistics(real_df, sid_col=sid_col, exclude_cols=exclude_cols)
    print("Computed statistics for the real dataset.")

    # Prepare a dictionary to store results
    msas_scores = {}

    # Iterate through synthetic datasets in the directory
    for filename in os.listdir(synthetic_dataset_dir):
        if filename.endswith(".csv"):  # Process only CSV files
            synthetic_path = os.path.join(synthetic_dataset_dir, filename)

            # Load the synthetic dataset
            synthetic_df = pd.read_csv(synthetic_path)

            # Compute statistics for the synthetic dataset
            synthetic_stats = compute_statistics(synthetic_df, sid_col=sid_col, exclude_cols=exclude_cols)

            # Compute the MSAS score
            msas_score = compute_msas(real_stats, synthetic_stats)

            # Extract epoch number and store the score
            epoch = extract_epoch_number(filename)
            msas_scores[epoch] = msas_score

            print(f"Computed MSAS score for {filename}: {msas_score:.4f}")

    # Write results to CSV
    output_df = pd.DataFrame.from_dict(msas_scores, orient='index', columns=['MSAS'])
    output_df.index.name = 'Epoch'
    output_df.reset_index(inplace=True)
    output_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    real_dataset = "../Row Data/Normalized_short.csv"  # Path to the real dataset
    synthetic_dataset_dir = "../Synth/"  # Path to the directory containing synthetic dataset CSV files
    output_csv = "msas_results.csv"  # Output CSV file to save MSAS scores

    msas_pipeline(real_dataset, synthetic_dataset_dir, output_csv)
