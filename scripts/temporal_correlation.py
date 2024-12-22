import os
import numpy as np
import pandas as pd
from scipy.fft import fft


class TemporalCorrelation:
    def __init__(self, real_csv_path, synthetic_dir_path, sequence_id_col, channel_cols, output_dir):
        """
        Initialize the analyzer with file paths and dataset specifications.

        Args:
            real_csv_path (str): Path to the real dataset CSV file.
            synthetic_dir_path (str): Path to the directory containing synthetic dataset CSV files.
            sequence_id_col (str): Column name identifying sequences in the dataset.
            channel_cols (list of str): Column names of the channels.
            output_dir (str): Path to the output CSV file.
        """
        self.real_csv_path = real_csv_path
        self.synthetic_dir_path = synthetic_dir_path
        self.sequence_id_col = sequence_id_col
        self.channel_cols = channel_cols
        self.output_dir = output_dir

    def read_multisequence_csv(self, file_path):
        """
        Reads a multivariate time series dataset with multiple sequences from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            list of numpy.ndarray: List of sequences, each as a 2D array (num_samples, num_channels).
        """
        try:
            df = pd.read_csv(file_path)
            sequences = []
            for _, group in df.groupby(self.sequence_id_col):
                sequence_data = group[self.channel_cols].to_numpy()
                sequences.append(sequence_data)
            return sequences
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except KeyError as e:
            raise KeyError(f"Column error: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading file {file_path}: {e}")

    def temporal_correlation_analysis(self, dataset, top_n_peaks):
        """
        Perform temporal correlation analysis on a dataset with multiple sequences.

        Args:
            dataset (list of numpy.ndarray): List of sequences, each of shape (num_samples, num_channels).
            top_n_peaks (int): Number of top peaks to consider.

        Returns:
            list of list of dict: Frequency components and normalized amplitudes for each sequence and channel.
        """
        all_results = []
        for sequence in dataset:
            sequence_results = []
            for channel in range(sequence.shape[1]):
                time_series = sequence[:, channel]

                # Perform FFT
                fft_result = fft(time_series)
                amplitudes = np.abs(fft_result)
                frequencies = np.fft.fftfreq(len(time_series))

                # Normalize amplitudes
                total_amplitude = np.sum(amplitudes)
                normalized_amplitudes = (
                    amplitudes / total_amplitude if total_amplitude > 0 else np.zeros_like(amplitudes)
                )

                # Remove DC component
                frequencies = frequencies[1:]
                normalized_amplitudes = normalized_amplitudes[1:]

                # Identify top N peaks
                peak_indices = np.argsort(normalized_amplitudes)[-top_n_peaks:]
                top_frequencies = frequencies[peak_indices]
                top_amplitudes = normalized_amplitudes[peak_indices]

                sequence_results.append({"frequencies": top_frequencies, "amplitudes": top_amplitudes})
            all_results.append(sequence_results)

        return all_results

    def squared_difference(self, real_peaks, synthetic_peaks):
        """
        Compute squared differences between real and synthetic datasets.

        Args:
            real_peaks (list of list of dict): Frequencies and amplitudes of real data.
            synthetic_peaks (list of list of dict): Frequencies and amplitudes of synthetic data.

        Returns:
            float: Average squared difference.
        """
        total_squared_diff = 0
        count = 0

        for real_sequence, synthetic_sequence in zip(real_peaks, synthetic_peaks):
            for real, synthetic in zip(real_sequence, synthetic_sequence):
                real_amplitudes = real["amplitudes"]
                synthetic_amplitudes = synthetic["amplitudes"]

                assert len(real_amplitudes) == len(synthetic_amplitudes), "Mismatch in peak counts."
                squared_diff = np.sum((real_amplitudes - synthetic_amplitudes) ** 2)
                total_squared_diff += squared_diff
                count += 1

        return total_squared_diff / count if count > 0 else 0

    def compute(self, top_n_peaks):
        """
        Process all synthetic files in the directory and compute Temporal Correlation scores.

        Args:
            top_n_peaks (int): Number of top peaks to consider.
        """
        try:
            # Read the real dataset
            real_dataset = self.read_multisequence_csv(self.real_csv_path)

            # Prepare results storage
            results = []

            # Process each synthetic file
            for synthetic_file in sorted(os.listdir(self.synthetic_dir_path)):
                if synthetic_file.endswith(".csv"):
                    epoch = os.path.splitext(synthetic_file)[0]  # Extract epoch from filename
                    synthetic_file_path = os.path.join(self.synthetic_dir_path, synthetic_file)

                    # Read the synthetic dataset
                    synthetic_dataset = self.read_multisequence_csv(synthetic_file_path)

                    # Ensure sequences are aligned
                    if len(real_dataset) != len(synthetic_dataset):
                        raise ValueError(f"Mismatch in number of sequences for file: {synthetic_file}")

                    # Perform temporal correlation analysis
                    real_peaks = self.temporal_correlation_analysis(real_dataset, top_n_peaks)
                    synthetic_peaks = self.temporal_correlation_analysis(synthetic_dataset, top_n_peaks)

                    # Compute squared difference (Temporal Correlation score)
                    avg_diff = self.squared_difference(real_peaks, synthetic_peaks)
                    results.append({"Epochs": epoch, "TC": avg_diff})

            # Save results to CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"{self.output_dir}/TC.csv", index=False)
            print(f"Results saved to {self.output_dir}")

        except Exception as e:
            raise RuntimeError(f"Error processing synthetic files: {e}")


# Example usage
if __name__ == "__main__":
    # File paths
    real_csv_path = "real_data.csv"
    synthetic_dir_path = "Synth"
    output_dir = "temporal_correlation_results.csv"

    # Column specifications
    sequence_id_col = "SID"
    channel_cols = [
        "Usage_kWh",
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
    ]

    # Initialize and run analysis
    analyzer = TemporalCorrelation(
        real_csv_path, synthetic_dir_path, sequence_id_col, channel_cols, output_dir
    )
    top_n_peaks = 5  # Number of peaks to consider
    try:
        analyzer.compute(top_n_peaks)
    except Exception as e:
        print(f"Analysis failed: {e}")
