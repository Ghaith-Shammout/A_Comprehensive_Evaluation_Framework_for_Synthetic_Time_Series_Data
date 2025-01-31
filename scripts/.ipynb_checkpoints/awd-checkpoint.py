import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


class AWD:
    """
    A class to compute and analyze Average Wasserstein Distance (AWD) between real and synthetic datasets.
    """

    def __init__(self, real_file: str, synth_dir: str, synth_dir_names: list[str], output_folder: str, exclude_columns: list[str] = None):
        """
        Initialize the AWD class.

        Parameters:
        - real_file (str): Path to the real data CSV file.
        - synth_dir (str): Path to the folder containing synthetic data CSV files.
        - output_folder (str): Directory to save AWD results and plots.
        - exclude_columns (list[str], optional): Columns to exclude from AWD computation. Default is None.
        """
        self.real_file = self._validate_file(real_file)
        self.synth_dir = self._validate_directory(synth_dir)
        self.synth_dir_names = synth_dir_names
        self.output_folder = self._create_output_directory(output_folder)
        self.exclude_columns = exclude_columns or []

    
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


    def compute(self) -> pd.DataFrame:
        """
        Compute AWD between the real data and synthetic datasets.
    
        Returns:
        - pd.DataFrame: DataFrame containing Epochs and AWD scores.
        """
        print("[+] Starting AWD computation...")
        real_df = pd.read_csv(self.real_file).drop(columns=self.exclude_columns, errors='ignore')
        
        for folder in self.synth_dir_names:
            results = []
            folder = str(folder)
            synth_folder = os.path.join(self.synth_dir, folder)
            print(f"[+] Processing synthetic folder: {synth_folder}")
            for synth_file in filter(lambda f: f.endswith('.csv'), os.listdir(synth_folder)):
                synth_path = os.path.join(synth_folder, synth_file)
                print(f"[+] Processing {synth_file}...")
                synth_data = pd.read_csv(synth_path).drop(columns=self.exclude_columns, errors='ignore')
                # Compute Wasserstein distances
                wd_scores = [
                    wasserstein_distance(real_df.iloc[:, j], synth_data.iloc[:, j])
                    for j in range(real_df.shape[1])
                ]
                avg_wd_score = np.mean(wd_scores)
                copy = int(synth_file.split('.')[0])
                print(f"[+] Copy {copy}, AVG(AWD) score: {avg_wd_score:.4f}")
                results.append({'Copy': copy, 'AWD': avg_wd_score})
            # Save results to a CSV file
            results_df = pd.DataFrame(results).sort_values(by="Copy").reset_index(drop=True)
            output_file = os.path.join(self.output_folder, "AWD", f"{folder}.csv")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the directory exists
            results_df.to_csv(output_file, index=False)
            print(f"[+] AWD results saved to '{output_file}'.")
        return results_df

    
    @staticmethod
    def plot_awd(wd_values_dict: dict, output_dir: str, awd_df: pd.DataFrame):
        """
        Plot AWD scores across epochs.

        Parameters:
        - wd_values_dict (dict): Dictionary with column names as keys and WD scores as values.
        - output_dir (str): Directory to save the plot.
        - awd_df (pd.DataFrame): DataFrame with epochs and AWD scores.
        """
        try:
            sorted_df = awd_df.sort_values(by="Epoch")
            sorted_epochs = sorted_df['Epoch'].values

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