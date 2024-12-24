import os
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

class AWD:
    def __init__(self, real_data, synth_folder, output_path, exclude_cols=None):
        """
        Initializes the AWDCalculator class.

        Parameters:
        - real_data: str, path to the real data CSV file.
        - synth_folder: str, path to the folder containing synthetic data CSV files.
        - exclude_cols: list, columns to exclude from both real and synthetic data (default is None).
        """
        self.real_data = real_data
        self.synth_folder = synth_folder
        self.exclude_cols = exclude_cols if exclude_cols is not None else []
        self.output_path = output_path

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
            epoch = int(file_name.split('.')[0])  # Get the part of the filename before the extension
            return epoch  # Return the extracted epoch part of the filename
        except ValueError:
            # If splitting or extracting fails (e.g., malformed filename), handle the error
            print(f"[-] Failed to extract epoch from filename: {file_name}")
            return -1  # Return -1 to indicate failure in extracting the epoch

    def compute(self, ):
        """
        Executes the AWD algorithm for the real and synthetic datasets and saves results to a CSV file.
        
        Parameters:
        
        
        Returns:
        - awd_df: pd.DataFrame, the dataframe containing AWD results.
        """
        print(f"[+] AWD calculation Started")
        
        # List to store the results for each file processed
        results = []

        try:
            # Step 1: Read the real data
            real = pd.read_csv(self.real_data)
            real_data = real.copy()
        except FileNotFoundError:
            print(f"[-] Error: The real data file {self.real_data} was not found.")
            return None
        except Exception as e:
            print(f"[-] Error reading real data: {e}")
            return None
        
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

                    # Step 7: Calculate the average Wasserstein Distance for the entire dataset
                    avg_wd_score = np.mean(wd_scores) if wd_scores else np.nan  # Average WD score (or NaN if no scores)

                    # Step 8: Extract the epoch number from the file name
                    epoch = self.extract_epoch_number(synth_file)  # Extract epoch number from the file name

                    # Store the result (epoch and AWD score)
                    results.append({'Epochs': epoch, 'AWD': avg_wd_score})

                except pd.errors.EmptyDataError:
                    print(f"[-] Error: {synth_file} is empty. Skipping this file.")
                except pd.errors.ParserError:
                    print(f"[-] Error parsing {synth_file}. Skipping this file.")
                except Exception as e:
                    print(f"[-] Error processing {synth_file}: {e}")

        # Step 9: Convert results into a pandas DataFrame
        awd_df = pd.DataFrame(results)

        # Step 10: Convert the 'Epochs' column to numeric, coercing errors (invalid values become NaN)
        awd_df['Epochs'] = pd.to_numeric(awd_df['Epochs'], errors='coerce')

        # Step 11: Sort the DataFrame by 'Epochs' and reset the index
        awd_df = awd_df.sort_values(by="Epochs").reset_index(drop=True)

        # Step 12: Save the final DataFrame to a CSV file
        try:
            awd_df.to_csv(f"{self.output_path}/AWD.csv", index=False)
            print(f"[+] AWD calculation completed & results saved to {self.output_path}/AWD.csv")
        except Exception as e:
            print(f"[-] Error saving AWD results to CSV: {e}")

        # Return the resulting DataFrame for further use
        return awd_df

    @staticmethod
    def plot_awd(wd_values_dict, synth_files, output_dir, awd_df):
        """
        Plot the Wasserstein Distance for each column over different synthetic datasets, sorted by epoch.
    
        Parameters:
        - wd_values_dict: dict, where keys are column names and values are lists of WD scores.
        - synth_files: list, names of synthetic data files.
        - awd_df: pandas DataFrame, containing the sorted results by Epochs.
        - output_dir: the directory to save the plot.
        """
        try:
            # Sort the synthetic files and WD values by the 'Epochs' column
            sorted_df = awd_df.sort_values(by="Epochs")  # Sort by Epochs
            sorted_epochs = sorted_df['Epochs'].values  # Get sorted epochs
            sorted_synth_files = [synth_files[i] for i in sorted_df.index]  # Corresponding filenames
            
            plt.figure(figsize=(14, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(wd_values_dict)))
    
            for idx, (column, wd_values) in enumerate(wd_values_dict.items()):
                # Sort the WD values by the sorted epochs
                sorted_wd_values = [wd_values[i] for i in sorted_df.index]
                plt.plot(sorted_epochs, sorted_wd_values, marker='o', label=column, color=colors[idx], markersize=6, linewidth=2)
    
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Wasserstein Distance (WD)', fontsize=12)
            plt.title('Evolution of Wasserstein Distance for Different Columns', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(sorted_epochs, sorted_synth_files, rotation=45, ha='right', fontsize=10)
            plt.legend(title="Columns", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.tight_layout()
    
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'awd.png'), bbox_inches='tight')
    
            print(f"[+] AWD plot saved to {output_dir}/awd.png")
    
        except Exception as e:
            print(f"[-] Error in plotting AWD: {e}")
