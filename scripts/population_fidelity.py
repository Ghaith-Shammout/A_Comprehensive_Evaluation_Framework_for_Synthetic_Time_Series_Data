import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
import os
import glob
from msas import MSAS


class PopulationFidelity:
    def __init__(self, real_data_path, synth_folder, exclude_cols):
        """
        Initializes the Evaluation class.

        Parameters:
        - real_data_path (str): Path to the real data CSV file.
        - synth_data_dir (str): Directory containing the synthetic data CSV files.
        - exclude_cols   (list): List excluded columns in calculating the evaluation measures.  
        """
        try:
            self.exclude_cols = exclude_cols
        except Exception as e:
            print(f"[-] Error Identifying excluded columns {exclude_cols}: {e}")
            raise
            
        try:
            # Load the real data
            self.real_data = real_data_path
        except Exception as e:
            print(f"[-] Error loading real data from {real_data_path}: {e}")
            raise
        
        try:
            # Get the list of synthetic CSV files in the directory
            self.synth_folder = synth_folder  # Adjust path as necessary
        except Exception as e:
            print(f"[-] Error reading synthetic data files from {synth_folder}: {e}")
            raise

    def compute_msas(self, output_file, x_step):
        
        msas = MSAS(real_file=self.real_data, synth_dir=self.synth_folder,
                    output_folder=output_file, x_step=x_step, exclude_columns=self.exclude_cols)
        msas.compute()
        
        
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
            # Attempt to extract the epoch from the filename by splitting it at the period (.)
            # This assumes the filename is in the format 'Epoch.csv', where 'Epoch' is a number.
            epoch = file_name.split('.')[0]  # Get the part of the filename before the extension
            return epoch  # Return the extracted epoch part of the filename
        except ValueError:
            # If splitting or extracting fails (e.g., malformed filename), handle the error
            print(f"[-] Failed to extract epoch from filename: {file_name}")
            return -1  # Return -1 to indicate failure in extracting the epoch


    

    
    """
    def awd(self, output_awd_csv):
        # Executes the AWD algorithm for the real and synthetic datasets and saves results to a CSV file.
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
    """
    
    def awd(self, output_awd_csv):
        """
        Executes the AWD algorithm for the real and synthetic datasets and saves results to a CSV file.
        This version does not process the data in sequences.
        """
        print(f"[+] AWD calculation Started")
        
        # List to store the results for each file processed
        results = []

        real = pd.read_csv(self.real_data)
        
        # Step 1: Read the real data
        real_data = real.copy()
    
        # Step 2: If there are columns to exclude, drop them from both synthetic and real data
        if self.exclude_cols:
            real_data = real_data.drop(columns=self.exclude_cols, errors='ignore')
    
        # Iterate over each synthetic data file
        for synth_file in os.listdir(self.synth_folder):
            if synth_file.endswith(".csv"):
                synth_path = os.path.join(self.synth_folder, synth_file)
                print(f"[+] Processing file: {synth_file}")
                try:
                    # Step 3: Read the synthetic data CSV file into a pandas DataFrame
                    synth_data = pd.read_csv(synth_path)
                    
                    # Step 4: If there are columns to exclude, drop them from synthetic data
                    if self.exclude_cols:
                        synth_data = synth_data.drop(columns=self.exclude_cols, errors='ignore')
        
                    # Step 5: Check if the number of rows and columns match
                    if real_data.shape[1] != synth_data.shape[1]:
                        print(f"[-] Number of columns in real data and synthetic data do not match. Skipping file: {synth_file}")
                        continue
        
                    # Step 6: Calculate the Wasserstein Distance for each column (feature)
                    wd_scores = []
                    for j in range(real_data.shape[1]):  # Iterate over columns
                        wd_score = wasserstein_distance(real_data.iloc[:, j], synth_data.iloc[:, j])
                        wd_scores.append(wd_score)
                        # Optional: print(f"Calculating wd_score for column {j}")  # Debug print for each calculation
        
                    # Step 7: Calculate the average Wasserstein Distance for the entire dataset
                    avg_wd_score = np.mean(wd_scores) if wd_scores else np.nan  # Average WD score (or NaN if no scores)
        
                    # Step 8: Extract the epoch number from the file name
                    file_name = os.path.basename(synth_file)  # Get only the file name, not the full path
                    epoch = self.extract_epoch_number(file_name)  # Extract epoch number from the file name
        
                    # Store the result (epoch and AWD score)
                    results.append({'Epochs': epoch, 'AWD': avg_wd_score})
                
                except Exception as e:
                    # If an error occurs while processing the file, print the error message
                    print(f"[-] Error processing {synth_file}: {e}")
    
        # Step 9: Convert results into a pandas DataFrame
        awd_df = pd.DataFrame(results)
    
        # Step 10: Convert the 'Epochs' column to numeric, coercing errors (invalid values become NaN)
        awd_df['Epochs'] = pd.to_numeric(awd_df['Epochs'], errors='coerce')
    
        # Step 11: Sort the DataFrame by 'Epochs' and reset the index
        awd_df = awd_df.sort_values(by="Epochs").reset_index(drop=True)
    
        # Step 12: Save the final DataFrame to a CSV file
        awd_df.to_csv(f"{output_awd_csv}/AWD.csv", index=False)
    
        # Step 13: Print confirmation that the process is complete
        print(f"[+] AWD calculation completed & results saved to {output_awd_csv}/awd.csv")
    
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
