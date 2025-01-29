import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp



class MSAS:
    """
    A class to compute the MSAS (Mean Statistic Alignment Score) between real and synthetic data.
    """

    def __init__(self, real_file: str, synth_dir: str, synth_dir_names: list[str], output_folder: str, exclude_columns: list[str] = None):
        """
        Initialize the MSAS class.

        Args:
            real_file (str): Path to the real data file.
            synth_dir (str): Path to the directory containing synthetic data files.
            synth_dir_names (list) : list of syntheic folders
            output_folder (str): Path to save the results CSV file.
            exclude_columns (list[str], optional): Columns to exclude from computation. Default is None.
        """
        self.real_file = real_file
        self.synth_dir = synth_dir
        self.synth_dir_names = synth_dir_names
        self.output_folder = output_folder
        self.exclude_columns = exclude_columns or []

    
    def preprocess_data(self, df, skip_columns):
        """Convert DataFrame with SID column to 3D array of sequences"""
        sequences = []
        for sid in df['SID'].unique():
            seq = df[df['SID'] == sid].drop(columns=skip_columns).values
            sequences.append(seq)
        return np.array(sequences)
    
    def compute_column_stats(self, column_data):
        stats = {}
        T = len(column_data)
        
        # Basic statistics
        stats['sequence_length'] = T
        stats['mean'] = np.mean(column_data)
        stats['median'] = np.median(column_data)
        stats['std_dev'] = np.std(column_data)
        
        # Inter-row differences
        stats['diff_1'] = np.mean(np.abs(np.diff(column_data, n=1))) if T >= 2 else 0.0
        stats['diff_5'] = np.mean(np.abs(column_data[5:] - column_data[:-5])) if T >= 6 else 0.0
        
        # Average difference across all steps
        total_diff = 0.0
        valid_steps = 0
        for x in range(1, T):
            if x < len(column_data):
                diffs = np.abs(column_data[x:] - column_data[:-x])
                if len(diffs) > 0:
                    total_diff += np.mean(diffs)
                    valid_steps += 1
        stats['diff_avg'] = total_diff / valid_steps if valid_steps > 0 else 0.0
        
        return stats
    
    def compute_msas(self, real_df, synthetic_df, skip_columns=None):
        """Main MSAS computation function with DataFrame inputs"""        
        # Preprocess data
        real_sequences = self.preprocess_data(real_df, skip_columns)
        synth_sequences = self.preprocess_data(synthetic_df, skip_columns)
        
        # Check dimensions
        if real_sequences.shape[2] != synth_sequences.shape[2]:
            raise ValueError("Mismatch in number of features after preprocessing")
        
        num_columns = real_sequences.shape[2]
        column_scores = []
        
        for c in range(num_columns):
            real_stats = {k: [] for k in ['sequence_length', 'mean', 'median', 'std_dev', 'diff_1', 'diff_5', 'diff_avg']}
            synth_stats = {k: [] for k in ['sequence_length', 'mean', 'median', 'std_dev', 'diff_1', 'diff_5', 'diff_avg']}
            
            # Collect statistics for real data
            for seq in real_sequences:
                col_data = seq[:, c]
                stats = self.compute_column_stats(col_data)
                for key in real_stats:
                    real_stats[key].append(stats[key])
            
            # Collect statistics for synthetic data
            for seq in synth_sequences:
                col_data = seq[:, c]
                stats = self.compute_column_stats(col_data)
                for key in synth_stats:
                    synth_stats[key].append(stats[key])
            
            # Compute KS scores
            ks_scores = []
            for stat in real_stats:
                ks_stat, _ = ks_2samp(real_stats[stat], synth_stats[stat])
                ks_scores.append(1 - ks_stat)
            
            column_score = np.mean(ks_scores)
            column_scores.append(column_score)
        
        return np.mean(column_scores)

    def compute(self):
        """
        Compute MSAS scores for synthetic data and save the results to a CSV file.
        """
        print("[+] Starting MSAS computation...")
        real_data = pd.read_csv(self.real_file)

        for folder in self.synth_dir_names:
            synth_folder = Path(f"./{self.synth_dir}/{folder}")
            results = []
            for synth_file in filter(lambda f: f.endswith('.csv'), os.listdir(synth_folder)):
                synth_path = os.path.join(synth_folder, synth_file)
                print(f"[+] Processing {synth_file}...")
                synth_data = pd.read_csv(synth_path)
                msas_score = self.compute_msas(real_data, synth_data, self.exclude_columns)
                copy = int(synth_file.split('.')[0])
                print(f"[+] Copy {copy}, MSAS score: {msas_score:.4f}")
                results.append({'Copy': copy, 'MSAS': msas_score})
            # Save results to a CSV file
            results_df = pd.DataFrame(results).sort_values(by="Copy").reset_index(drop=True)
            output_file = os.path.join(self.output_folder, "MSAS", f"{folder}.csv")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the directory exists
            results_df.to_csv(output_file, index=False)
            print(f"[+] MSAS results saved to '{output_file}'.")