import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
import os
import glob


class PopulationFidelity:
    def __init__(self, real_data_path, synth_folder, exclude_cols, sequence_length=1):
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

    @staticmethod
    def extract_epoch_number(filename):
        """
        Extracts the epoch number from the filename. Assumes the filename contains the word 'epoch' followed by a number.
        Example: synthetic_data_epoch1.csv -> epoch1
        """
        base_name = os.path.splitext(filename)[0]
        epoch_part = [part for part in base_name.split('_') if 'epoch' in part]
        return epoch_part[0] if epoch_part else base_name

    def msas(self, output_msas_csv):
        """
        Executes the MSAS algorithm for the real and synthetic datasets and saves results to a CSV file.
        """
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
        print(f"[+] MSAS results saved to {output_msas_csv}")
        return msas_df



    def get_most_likely_period(self, data):
        """
        Calculates the most likely period using FFT for a given data sequence.

        Parameters:
        - data (numpy array): Data sequence to process.

        Returns:
        - float: The most likely period.
        """
        try:
            n = len(data)
            fft_result = np.fft.fft(data)
            fft_amplitude = np.abs(fft_result)[:n // 2]  # Use only positive frequencies
            freqs = np.fft.fftfreq(n, d=1)[:n // 2]
            peak_freq = freqs[np.argmax(fft_amplitude)]  # Get the frequency with max amplitude
            return 1 / peak_freq if peak_freq != 0 else np.inf  # Return the period
        except Exception as e:
            print(f"[-] Error calculating the most likely period: {e}")
            raise

    def compare_amplitudes(self, real_data, synth_data):
        """
        Compares the FFT amplitudes using Wasserstein distance between real and synthetic data.

        Parameters:
        - real_data (numpy array): Real data sequence.
        - synth_data (numpy array): Synthetic data sequence.

        Returns:
        - float: The Wasserstein distance between the amplitudes.
        """
        try:
            real_amplitudes = np.abs(np.fft.fft(real_data))[:len(real_data) // 2]
            synth_amplitudes = np.abs(np.fft.fft(synth_data))[:len(synth_data) // 2]
            return wasserstein_distance(real_amplitudes, synth_amplitudes)
        except Exception as e:
            print(f"[-] Error comparing amplitudes: {e}")
            raise

    def awd(self, output_awd_csv):
        """
        Executes the AWD algorithm for the real and synthetic datasets and saves results to a CSV file.
        """
        sequence_results = []
    
        for synth_file in self.synth_data_files:
            print(f"[+] Processing file: {synth_file}")
            try:
                synth_data = pd.read_csv(synth_file)
                synth_data = synth_data.drop(columns=self.exclude_cols)
    
                file_wd_scores = []
                for i in range(0, len(self.real_data), self.sequence_length):
                    real_sequence = self.real_data.iloc[i:i + self.sequence_length].values
                    synth_sequence = synth_data.iloc[i:i + self.sequence_length].values
    
                    if real_sequence.shape[0] == self.sequence_length and synth_sequence.shape[0] == self.sequence_length:
                        wd_scores = [
                            self.compare_amplitudes(real_sequence[:, i], synth_sequence[:, i])
                            for i in range(real_sequence.shape[1])
                        ]
                        avg_wd_score = np.mean(wd_scores) if wd_scores else np.nan
                        file_wd_scores.append(avg_wd_score)
    
                if file_wd_scores:
                    file_avg_wd_score = np.mean(file_wd_scores)
                    epoch = self.extract_epoch_number(os.path.basename(synth_file))
                    sequence_results.append({'Epochs': epoch, 'AWD': file_avg_wd_score})
            except Exception as e:
                print(f"[-] Error processing {synth_file}: {e}")

        # Create DataFrame and sort by Epochs (convert 'Epochs' to numeric before sorting)
        awd_df = pd.DataFrame(sequence_results)
        awd_df['Epochs'] = pd.to_numeric(awd_df['Epochs'], errors='coerce')  # Convert 'Epochs' to numeric
        awd_df = awd_df.sort_values(by="Epochs").reset_index(drop=True)
        
        # Save the sorted DataFrame to CSV
        awd_df.to_csv(output_awd_csv, index=False)
        print(f"[+] AWD results saved to {output_awd_csv}")
        
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
    parser.add_argument("--exclude_cols", required=False, help="List of columns to exclude in calculating evaluation measures")
    parser.add_argument("--seq_len", required=False, help="Length of every sequence (default = 1)")
    args = parser.parse_args()

