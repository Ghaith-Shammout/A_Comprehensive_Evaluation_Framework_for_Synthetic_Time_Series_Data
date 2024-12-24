import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
import os
import glob
from msas import MSAS
from temporal_correlation import TemporalCorrelation
from awd import AWD


class PopulationFidelity:
    def __init__(self, real_data_path, synth_folder, output_folder):
        """
        Initializes the Evaluation class.

        Parameters:
        - real_data_path (str): Path to the real data CSV file.
        - synth_data_dir (str): Directory containing the synthetic data CSV files.
        - exclude_cols   (list): List excluded columns in calculating the evaluation measures.  
        """           
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
        
        try:
            self.output_folder = output_folder
        except Exception as e:
            print(f"[-] Error Identifying excluded columns {exclude_cols}: {e}")
            raise


    def compute_msas(self, x_step, exclude_columns):
        
        msas = MSAS(real_file=self.real_data, synth_dir=self.synth_folder,
                    output_folder=self.output_folder, x_step=x_step, exclude_columns=exclude_columns)
        msas.compute()

    def compute_temp_corr(self, seq_id, channel_cols, top_peaks):
        # Initialize and run analysis
        temp_corr = TemporalCorrelation(real_csv_path=self.real_data,
                                       synthetic_dir_path=self.synth_folder,
                                       sequence_id_col=seq_id, 
                                       channel_cols=channel_cols,
                                       output_file=self.output_folder)
        top_n_peaks = top_peaks  # Number of peaks to consider
        try:
            temp_corr.compute(top_n_peaks)
        except Exception as e:
            print(f"Analysis failed: {e}")

    def compute_awd(self, exclude_columns, plot_output_path):
        # Initialize & Run AWD
        awd = AWD(real_data=self.real_data,
                  synth_folder=self.synth_folder,
                  output_path=self.output_folder,
                  exclude_cols=exclude_columns)
        awd_df = awd.compute()    
        if awd_df is not None:
            # Optionally, plot the AWD results if the calculation was successful
            wd_values_dict = {col: awd_df['AWD'].values for col in awd_df.columns}
            awd.plot_awd(wd_values_dict, os.listdir(self.synth_folder), plot_output_path, awd_df)
        
        
        
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation metrics for synthetic data.")
    parser.add_argument("--real_data_path", required=True, help="Path to the real data CSV file.")
    parser.add_argument("--synth_folder", required=True, help="Path to the folder containing synthetic data CSV files.")
    parser.add_argument("--exclude_cols", required=False, help="List of columns to exclude in calculating evaluation measures", default=[])
    parser.add_argument("--seq_len", required=False, type=int, help="Length of every sequence (default = 1)", default=1)
    parser.add_argument("--output_awd_csv", required=True, help="Path to save the AWD output CSV")
    args = parser.parse_args()
