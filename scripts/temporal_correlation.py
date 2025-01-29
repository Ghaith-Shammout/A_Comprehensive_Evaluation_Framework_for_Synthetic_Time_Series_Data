import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.fft import fft


class TemporalCorrelation:
    """
    A class to analyze temporal correlation between real and synthetic datasets using FFT.
    """

    def __init__(self, real_file: str, synth_dir: str, synth_dir_names: list[str], output_folder: str, channel_cols: list[str]):
        """
        Initialize the TemporalCorrelation class.

        Args:
            real_file (str): Path to the real dataset CSV file.
            synth_dir (str): Path to the directory containing synthetic dataset CSV files.
            synth_dir_names : list of synthetic directory names
            output_folde (str) : Directory to save output results.
            channel_cols (list[str]): Column names of the channels.
        """
        self.real_file = self._validate_file(real_file)
        self.synth_dir = self._validate_directory(synth_dir)
        self.synth_dir_names = synth_dir_names
        self.output_folder = self._create_output_directory(output_folder)
        self.channel_cols = channel_cols

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
        """Create the output directory if it does not exist."""
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def read_multisequence_csv(self, file_path: str) -> list[np.ndarray]:
        """
        Read a multivariate time series dataset with multiple sequences from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            list[np.ndarray]: List of sequences, each as a 2D array (num_samples, num_channels).
        """
        try:
            df = pd.read_csv(file_path)
            sequences = [
                group[self.channel_cols].to_numpy()
                for _, group in df.groupby("SID")
            ]
            return sequences
        except Exception as e:
            raise RuntimeError(f"Error reading file {file_path}: {e}")

    def temporal_correlation_analysis(self, dataset: list[np.ndarray], top_n_peaks: int) -> list[list[dict]]:
        """
        Perform temporal correlation analysis on a dataset with multiple sequences.

        Args:
            dataset (list[np.ndarray]): List of sequences, each of shape (num_samples, num_channels).
            top_n_peaks (int): Number of top peaks to consider.

        Returns:
            list[list[dict]]: Frequency components and normalized amplitudes for each sequence and channel.
        """
        all_results = []
        for sequence in dataset:
            sequence_results = []
            for channel in range(sequence.shape[1]):
                time_series = sequence[:, channel]
                fft_result = fft(time_series)
                amplitudes = np.abs(fft_result)
                frequencies = np.fft.fftfreq(len(time_series))

                total_amplitude = np.sum(amplitudes)
                normalized_amplitudes = (
                    amplitudes / total_amplitude if total_amplitude > 0 else np.zeros_like(amplitudes)
                )

                # Remove DC component
                frequencies = frequencies[1:]
                normalized_amplitudes = normalized_amplitudes[1:]

                # Identify top N peaks
                peak_indices = np.argsort(normalized_amplitudes)[-top_n_peaks:]
                sequence_results.append({
                    "frequencies": frequencies[peak_indices],
                    "amplitudes": normalized_amplitudes[peak_indices]
                })
            all_results.append(sequence_results)
        return all_results

    @staticmethod
    def squared_difference(real_peaks: list[list[dict]], synthetic_peaks: list[list[dict]]) -> float:
        """
        Compute squared differences between real and synthetic datasets.

        Args:
            real_peaks (list[list[dict]]): Frequencies and amplitudes of real data.
            synthetic_peaks (list[list[dict]]): Frequencies and amplitudes of synthetic data.

        Returns:
            float: Average squared difference.
        """
        total_squared_diff = 0
        count = 0
        for real_seq, synth_seq in zip(real_peaks, synthetic_peaks):
            for real, synth in zip(real_seq, synth_seq):
                assert len(real["amplitudes"]) == len(synth["amplitudes"]), "Mismatch in peak counts."
                total_squared_diff += np.sum((real["amplitudes"] - synth["amplitudes"]) ** 2)
                count += 1
        return total_squared_diff / count if count > 0 else 0

    def compute(self):
        """
        Process all synthetic files and compute Temporal Correlation scores.

        Args:
            top_n_peaks (int): Number of top peaks to consider.
        """
        top_n_peaks = 5 
        print("[+] Starting Temporal Correlation computation...")
        real_dataset = self.read_multisequence_csv(self.real_file)
        for folder in self.synth_dir_names:
            synth_folder = Path(f"./{self.synth_dir}/{folder}")
            results = []
            for synth_file in filter(lambda f: f.endswith('.csv'), os.listdir(synth_folder)):
                synth_path = os.path.join(synth_folder, synth_file)
                print(f"[+] Processing {synth_file}...")
                synthetic_dataset = self.read_multisequence_csv(synth_path)
                if len(real_dataset) != len(synthetic_dataset):
                    raise ValueError(f"Mismatch in sequence counts for file: {synthetic_file}")
                real_peaks = self.temporal_correlation_analysis(real_dataset, top_n_peaks)
                synthetic_peaks = self.temporal_correlation_analysis(synthetic_dataset, top_n_peaks)
                avg_diff = self.squared_difference(real_peaks, synthetic_peaks)
                copy = int(synth_file.split('.')[0])
                print(f"[+] Copy {copy}, TC score: {avg_diff:.4f}")
                results.append({'Copy': copy, 'TC': avg_diff})
            # Save results to a CSV file
            results_df = pd.DataFrame(results).sort_values(by="Copy").reset_index(drop=True)
            output_file = os.path.join(self.output_folder, "TC", f"{folder}.csv")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the directory exists
            results_df.to_csv(output_file, index=False)
            print(f"[+] TC results saved to '{output_file}'.")
            
                
                
