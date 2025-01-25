import os
import pandas as pd
from msas import MSAS
from temporal_correlation import TemporalCorrelation
from awd import AWD


class PopulationFidelity:
    """
    A class to evaluate population fidelity between real and synthetic data using MSAS, 
    Temporal Correlation, and AWD metrics.
    """

    def __init__(self, real_data_path: str, synth_folder: str, output_folder: str):
        """
        Initialize the PopulationFidelity class.

        Parameters:
        - real_data_path (str): Path to the real data CSV file.
        - synth_folder (str): Directory containing synthetic data CSV files.
        - output_folder (str): Directory to save output results.
        """
        self.real_data = self._load_real_data(real_data_path)
        self.synth_folder = self._validate_directory(synth_folder)
        self.output_folder = self._create_output_directory(output_folder)

    @staticmethod
    def _load_real_data(real_data_path: str) -> str:
        """Load and validate the real data file."""
        if not os.path.isfile(real_data_path):
            raise FileNotFoundError(f"Real data file '{real_data_path}' does not exist.")
        return real_data_path

    @staticmethod
    def _validate_directory(directory_path: str) -> str:
        """Validate that the provided directory exists."""
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")
        return directory_path

    @staticmethod
    def _create_output_directory(output_folder: str) -> str:
        """Create the output directory if it doesn't already exist."""
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def compute_msas(self, x_step: int, exclude_columns: list[str]):
        """
        Compute MSAS evaluation.

        Parameters:
        - x_step (int): Step size for X-axis in MSAS plots.
        - exclude_columns (list[str]): Columns to exclude from MSAS computation.
        """
        msas = MSAS(
            real_file=self.real_data,
            synth_dir=self.synth_folder,
            output_folder=self.output_folder,
            x_step=x_step,
            exclude_columns=exclude_columns,
        )
        msas.compute()

    def compute_temporal_correlation(self, sequence_id: str, channel_columns: list[str], top_peaks: int):
        """
        Compute temporal correlation analysis.

        Parameters:
        - sequence_id (str): Column identifying the sequence in the dataset.
        - channel_columns (list[str]): List of channel columns to analyze.
        - top_peaks (int): Number of top peaks to consider in correlation analysis.
        """
        temp_corr = TemporalCorrelation(
            real_csv_path=self.real_data,
            synthetic_dir_path=self.synth_folder,
            sequence_id_col=sequence_id,
            channel_cols=channel_columns,
            output_dir=self.output_folder,
        )
        try:
            temp_corr.compute(top_peaks)
        except Exception as e:
            print(f"[-] Error in Temporal Correlation analysis: {e}")

    def compute_awd(self, exclude_columns: list[str], plot_output_path: str):
        """
        Compute AWD evaluation and optionally generate plots.

        Parameters:
        - exclude_columns (list[str]): Columns to exclude from AWD computation.
        - plot_output_path (str): Path to save AWD plots.
        """
        awd = AWD(
            real_data=self.real_data,
            synth_folder=self.synth_folder,
            output_path=self.output_folder,
            exclude_cols=exclude_columns,
        )
        try:
            awd_df = awd.compute()
            if awd_df is not None:
                wd_values_dict = {col: awd_df['AWD'].values for col in awd_df.columns}
                #awd.plot_awd(wd_values_dict, os.listdir(self.synth_folder), plot_output_path, awd_df)
        except Exception as e:
            print(f"[-] Error in AWD computation: {e}")

