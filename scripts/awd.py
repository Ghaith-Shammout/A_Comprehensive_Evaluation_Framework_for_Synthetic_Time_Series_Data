import os
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt


class AWD:
    """
    A class to compute and analyze Average Wasserstein Distance (AWD) between real and synthetic datasets.
    """

    def __init__(self, real_data: str, synth_folder: str, output_path: str, exclude_cols: list[str] = None):
        """
        Initialize the AWD class.

        Parameters:
        - real_data (str): Path to the real data CSV file.
        - synth_folder (str): Path to the folder containing synthetic data CSV files.
        - output_path (str): Directory to save AWD results and plots.
        - exclude_cols (list[str], optional): Columns to exclude from AWD computation. Default is None.
        """
        self.real_data = self._validate_file(real_data)
        self.synth_folder = self._validate_directory(synth_folder)
        self.output_path = self._create_output_directory(output_path)
        self.exclude_cols = exclude_cols or []

    
    @staticmethod
    def _validate_file(file_path: str) -> str:
        """Validate that the file exists."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")
        return file_path

    @staticmethod
    def _validate_directory(directory_path: str) -> str:
        """Validate that the directory exists."""
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")
        return directory_path

    @staticmethod
    def _create_output_directory(output_folder: str) -> str:
        """Create the output directory if it doesn't exist."""
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    @staticmethod
    def extract_epoch_number(file_name: str) -> int:
        """
        Extract the epoch number from the synthetic file name.

        Parameters:
        - file_name (str): Name of the synthetic file.

        Returns:
        - int: Extracted epoch number or -1 if extraction fails.
        """
        try:
            return int(file_name.split('.')[0])
        except ValueError:
            print(f"[-] Failed to extract epoch from filename: {file_name}")
            return -1

    def compute(self) -> pd.DataFrame:
        """
        Compute AWD between the real data and synthetic datasets.

        Returns:
        - pd.DataFrame: DataFrame containing Epochs and AWD scores.
        """
        print("[*] Starting AWD computation...")
        results = []

        # Load the real data
        try:
            real_data = pd.read_csv(self.real_data).drop(columns=self.exclude_cols, errors='ignore')
        except Exception as e:
            print(f"[-] Error loading real data: {e}")
            return None

        # Process each synthetic file
        for synth_file in filter(lambda f: f.endswith('.csv'), os.listdir(self.synth_folder)):
            synth_path = os.path.join(self.synth_folder, synth_file)
            try:
                synth_data = pd.read_csv(synth_path).drop(columns=self.exclude_cols, errors='ignore')

                # Check column alignment
                if real_data.shape[1] != synth_data.shape[1]:
                    print(f"[-] Column mismatch with {synth_file}. Skipping.")
                    continue

                # Compute Wasserstein distances
                wd_scores = [
                    wasserstein_distance(real_data.iloc[:, j], synth_data.iloc[:, j])
                    for j in range(real_data.shape[1])
                ]
                avg_wd_score = np.mean(wd_scores)

                # Extract epoch and append results
                copy = self.extract_epoch_number(synth_file)
                results.append({'Copy': copy, 'AWD': avg_wd_score})
                print(f"[+] Processed {synth_file} (Copy) {copy}): AWD={avg_wd_score}")

            except Exception as e:
                print(f"[-] Error processing {synth_file}: {e}")

        # Create results DataFrame
        awd_df = pd.DataFrame(results).sort_values(by="Copy").reset_index(drop=True)

        # Save results
        output_csv = os.path.join(self.output_path, "AWD", f"{os.path.basename(self.synth_folder)}.csv")
        try:
            awd_df.to_csv(output_csv, index=False)
            print(f"[+] AWD results saved to '{output_csv}'.")
        except Exception as e:
            print(f"[-] Error saving AWD results: {e}")

        return awd_df

    @staticmethod
    def plot_awd(wd_values_dict: dict, synth_files: list, output_dir: str, awd_df: pd.DataFrame):
        """
        Plot AWD scores across epochs.

        Parameters:
        - wd_values_dict (dict): Dictionary with column names as keys and WD scores as values.
        - synth_files (list): List of synthetic file names.
        - output_dir (str): Directory to save the plot.
        - awd_df (pd.DataFrame): DataFrame with epochs and AWD scores.
        """
        try:
            sorted_df = awd_df.sort_values(by="Epochs")
            sorted_epochs = sorted_df['Epochs'].values

            plt.figure(figsize=(14, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(wd_values_dict)))

            for idx, (column, wd_values) in enumerate(wd_values_dict.items()):
                plt.plot(sorted_epochs, wd_values, marker='o', label=column, color=colors[idx])

            plt.xlabel('Epochs')
            plt.ylabel('Wasserstein Distance (WD)')
            plt.title('AWD Evolution')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title="Columns", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'awd.png')
            plt.savefig(plot_path)
            print(f"[+] AWD plot saved to '{plot_path}'.")

        except Exception as e:
            print(f"[-] Error in plotting AWD: {e}")
