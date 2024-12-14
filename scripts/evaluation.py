import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
import os
import glob


class PopulationFidelity:
    def __init__(self, real_data_path, synth_folder, exclude_cols, sequence_length):
        """
        Initializes the Evaluation class.

        Parameters:
        - real_data_path (str): Path to the real data CSV file.
        - synth_data_dir (str): Directory containing the synthetic data CSV files.
        - exclude_cols   (list): List excluded columns in calculating the evaluation measures.  
        - sequence_length (int): Number of data points per sequence (default is 1).
        """
        try:
            self.exclude_cols = exclude_cols
        except Exception as e:
            print(f"[-] Error Identifying excluded columns {exclude_cols}: {e}")
            raise
            
        try:
            # Load the real data
            self.real_data = pd.read_csv(real_data_path)
            # Drop unnecessary columns
            self.real_data = self.real_data.drop(columns=exclude_cols)
        except Exception as e:
            print(f"[-] Error loading real data from {real_data_path}: {e}")
            raise
        
        try:
            # Get the list of synthetic CSV files in the directory
            self.synth_data_files = glob.glob(synth_folder + '/*.csv')  # Adjust path as necessary
        except Exception as e:
            print(f"[-] Error reading synthetic data files from {synth_folder}: {e}")
            raise
        

        try:
            self.sequence_length = sequence_length
        except Exception as e:
            print(f"[-] Error Identifying sequence length {sequence_length}: {e}")
            raise
        

    @staticmethod
    def compute_inter_row_dependency(sequence, lag=1):
        """
        Computes the average difference between a value in row n and the value `lag` steps after it in a sequence.
        """
        return np.mean(np.abs(sequence[:-lag] - sequence[lag:]))

    @staticmethod
    def compute_statistics(df, window_size=96):
        """
        Computes statistics for fixed-size windows across the dataset.
        Parameters:
            df (pd.DataFrame): Input DataFrame containing the time series data.
            window_size (int): Size of each window in data points.
            exclude_cols (list): Columns to exclude from statistical computation.
        Returns:
            pd.DataFrame: DataFrame containing statistics for each window.
        """
        stats = []
        columns = [col for col in df.columns]
        
        num_windows = len(df) // window_size  # Number of full windows
        for i in range(num_windows):
            window_start = i * window_size
            window_end = window_start + window_size
            window_data = df.iloc[window_start:window_end]
            
            window_stats = {'Window': i + 1}
            window_stats['length'] = len(window_data)
            
            for col in columns:
                data = window_data[col].values
                window_stats[f'{col}_mean'] = np.mean(data)
                window_stats[f'{col}_median'] = np.median(data)
                window_stats[f'{col}_std'] = np.std(data)
                window_stats[f'{col}_inter_row_dep'] = PopulationFidelity.compute_inter_row_dependency(data, lag=1)
            
            stats.append(window_stats)
        
        return pd.DataFrame(stats)

    @staticmethod
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


    def msas(self, output_msas_csv):
        """
        Executes the MSAS algorithm for the real and synthetic datasets and saves results to a CSV file.
        """
        print("[+] MSAS Calculation Started")
        
        real_stats = self.compute_statistics(self.real_data, window_size=self.sequence_length)
        print(f"[+] Completed processing real dataset")
        
        msas_scores = []
    
        for synth_file in self.synth_data_files:
            try:
                print(f"[+] Processing file: {synth_file}")
    
                synthetic_df = pd.read_csv(synth_file)
                synthetic_df = synthetic_df.drop(columns=self.exclude_cols)
    
                synthetic_stats = self.compute_statistics(synthetic_df, window_size=self.sequence_length)
                msas_score = self.compute_msas(real_stats, synthetic_stats)
    
                epoch = self.extract_epoch_number(os.path.basename(synth_file))
                msas_scores.append({'Epochs': epoch, 'MSAS': msas_score})
    
                # print(f"[+] Computed MSAS score for {epoch}: {msas_score:.4f}")
            except Exception as e:
                print(f"[-] Error processing {synth_file}: {e}")
    
        # Create DataFrame and sort by Epochs (convert 'Epochs' to numeric before sorting)
        msas_df = pd.DataFrame(msas_scores)
        msas_df['Epochs'] = pd.to_numeric(msas_df['Epochs'], errors='coerce')  # Convert 'Epochs' to numeric
        msas_df = msas_df.sort_values(by="Epochs").reset_index(drop=True)

        # Save the sorted DataFrame to CSV
        msas_df.to_csv(output_msas_csv, index=False)
        print(f"[+] MSAS calculation completed & results saved to {output_msas_csv}")
        return msas_df


    def extract_epoch_number(self, file_name):
        """
        Extracts the epoch number from the synthetic data file name.
        This method assumes that the filename is in the format 'Epochs.csv', e.g., '1000.csv'.

        Parameters:
            file_name (str): The name of the synthetic data file.

        Returns:
            int: The extracted epoch number or a default value (-1) if extraction fails.
        """
        try:
            # Attempt to extract epoch from the filename, assuming it's numeric before the extension
            epoch = file_name.split('.')[0]
            return epoch
        except ValueError:
            print(f"[-] Failed to extract epoch from filename: {file_name}")
            return -1  # Return a default value (e.g., -1) when epoch cannot be extracted

     
    def awd(self, output_awd_csv):
        """
        Executes the AWD algorithm for the real and synthetic datasets and saves results to a CSV file.
        """
        print(f"[+] AWD calculation Started")
    
        # List to store the results for each file processed
        sequence_results = []
    
        # Iterate over each synthetic data file
        for synth_file in self.synth_data_files:
            print(f"[+] Processing file: {synth_file}")
            try:
                # Step 1: Read the synthetic data CSV file into a pandas DataFrame
                synth_data = pd.read_csv(synth_file)
    
                # Step 2: If there are columns to exclude, drop them from both synthetic and real data
                if self.exclude_cols:
                    synth_data = synth_data.drop(columns=self.exclude_cols, errors='ignore')
                    self.real_data = self.real_data.drop(columns=self.exclude_cols, errors='ignore')
    
                # List to store the Wasserstein Distance (WD) scores for each sequence
                file_wd_scores = []
    
                # Step 3: Loop through the real data in chunks of 'sequence_length'
                for i in range(0, len(self.real_data), self.sequence_length):
                    # Extract a sequence from the real and synthetic data
                    real_sequence = self.real_data.iloc[i:i + self.sequence_length].values
                    synth_sequence = synth_data.iloc[i:i + self.sequence_length].values
    
                    # Step 4: If the sequence is too short (less than 'sequence_length' rows), skip it
                    if real_sequence.shape[0] != self.sequence_length or synth_sequence.shape[0] != self.sequence_length:
                        print(f"[-] Sequence at index {i} is too short (expected {self.sequence_length} rows). Skipping this sequence.")
                        continue
    
                    # Step 5: Calculate the Wasserstein Distance for each feature (column) of the sequence
                    wd_scores = []
                    for j in range(real_sequence.shape[1]):  # Iterate over columns
                        # Calculate WD for each column and append the result
                        wd_score = wasserstein_distance(real_sequence[:, j], synth_sequence[:, j])
                        wd_scores.append(wd_score)
                        #print(f"Calculating wd_score for seq:{i} & col:{j}")  # Debug print for each calculation
    
                    # Step 6: Calculate the average Wasserstein Distance for this sequence
                    avg_wd_score = np.mean(wd_scores) if wd_scores else np.nan  # Average WD score (or NaN if no scores)
                    file_wd_scores.append(avg_wd_score)  # Store the average WD score for this sequence
    
                # Step 7: If there are WD scores for this file, calculate the average AWD score
                if file_wd_scores:
                    file_avg_wd_score = np.mean(file_wd_scores)  # Average AWD for the current synthetic file
                    # Extract only the file name (not the full path) for epoch extraction
                    file_name = os.path.basename(synth_file)  # Get only the file name, not the full path
                    epoch = self.extract_epoch_number(file_name)  # Extract epoch number from the file name
                    # Store the result (epoch and AWD score)
                    sequence_results.append({'Epochs': epoch, 'AWD': file_avg_wd_score})
    
            except Exception as e:
                # If an error occurs while processing the file, print the error message
                print(f"[-] Error processing {synth_file}: {e}")
    
        # Step 8: Convert results into a pandas DataFrame
        awd_df = pd.DataFrame(sequence_results)
    
        # Step 9: Convert the 'Epochs' column to numeric, coercing errors (invalid values become NaN)
        awd_df['Epochs'] = pd.to_numeric(awd_df['Epochs'], errors='coerce')
    
        # Step 10: Sort the DataFrame by 'Epochs' and reset the index
        awd_df = awd_df.sort_values(by="Epochs").reset_index(drop=True)
    
        # Step 11: Save the final DataFrame to a CSV file
        awd_df.to_csv(output_awd_csv, index=False)
    
        # Step 12: Print confirmation that the process is complete
        print(f"[+] AWD calculation completed & results saved to {output_awd_csv}")
    
        # Return the resulting DataFrame for further use
        return awd_df


    
    @staticmethod
    def plot_awd(wd_values_dict, synth_files):
        """
        Plot the Wasserstein Distance for each column over different synthetic datasets.
        """
        plt.figure(figsize=(14, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(wd_values_dict)))

        for idx, (column, wd_values) in enumerate(wd_values_dict.items()):
            plt.plot(range(len(wd_values)), wd_values, marker='o', label=column, color=colors[idx], markersize=6, linewidth=2)

        plt.xlabel('Index of Synthetic Dataset', fontsize=12)
        plt.ylabel('Wasserstein Distance (WD)', fontsize=12)
        plt.title('Evolution of Wasserstein Distance for Different Columns', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(range(len(synth_files)), synth_files, rotation=45, ha='right', fontsize=10)
        plt.legend(title="Columns", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()

        output_dir = './outputs/Plots/'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'awd.png'), bbox_inches='tight')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation metrics for synthetic data.")
    parser.add_argument("--real_data_path", required=True, help="Path to the real data CSV file.")
    parser.add_argument("--synth_folder", required=True, help="Path to the folder containing synthetic data CSV files.")
    parser.add_argument("--exclude_cols", required=False, help="List of columns to exclude in calculating evaluation measures", default=[])
    parser.add_argument("--seq_len", required=False, type=int, help="Length of every sequence (default = 1)", default=1)
    parser.add_argument("--output_awd_csv", required=True, help="Path to save the AWD output CSV")
    args = parser.parse_args()

    # Instantiate the PopulationFidelity class
    pf = PopulationFidelity(
        real_data_path=args.real_data_path,
        synth_folder=args.synth_folder,
        exclude_cols=args.exclude_cols.split(',') if args.exclude_cols else [],
        sequence_length=args.seq_len
    )
    
    # Run only the AWD method
    awd_results = pf.awd(output_awd_csv=args.output_awd_csv)
    print(f"[+] AWD results saved to {args.output_awd_csv}")
