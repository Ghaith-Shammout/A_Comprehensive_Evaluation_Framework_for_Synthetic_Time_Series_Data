import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


class MSAS:
    """
    A class to compute the MSAS (Mean Statistic Alignment Score) between real and synthetic data.
    """

    def __init__(self, real_file: str, synth_dir: str, output_folder: str, x_step: int, exclude_columns: list[str] = None):
        """
        Initialize the MSAS class.

        Args:
            real_file (str): Path to the real data file.
            synth_dir (str): Path to the directory containing synthetic data files.
            output_folder (str): Path to save the results CSV file.
            x_step (int): Step size for inter-row dependency calculation.
            exclude_columns (list[str], optional): Columns to exclude from computation. Default is None.
        """
        self.real_file = self._validate_file(real_file)
        self.synth_dir = self._validate_directory(synth_dir)
        self.output_folder = self._create_output_directory(output_folder)
        self.x_step = x_step
        self.exclude_columns = exclude_columns or []
        self.real_data = pd.read_csv(self.real_file)
        self.statistics = ['mean', 'median', 'std', 'length', 'inter_row_dep']

    @staticmethod
    def _validate_file(file_path: str) -> str:
        """Validate that the specified file exists."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")
        return file_path

    @staticmethod
    def _validate_directory(directory_path: str) -> str:
        """Validate that the specified directory exists."""
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")
        return directory_path

    @staticmethod
    def _create_output_directory(output_folder: str) -> str:
        """Create the output directory if it does not exist."""
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def ks(self, real_data: pd.Series, synthetic_data: pd.Series) -> float:
        """
        Compute the Kolmogorov-Smirnov (KS) statistic for two distributions.

        Args:
            real_data (pd.Series): Real data column.
            synthetic_data (pd.Series): Synthetic data column.

        Returns:
            float: 1 minus the KS statistic, representing alignment (higher is better).
        """
        real_data = real_data.dropna()
        synthetic_data = synthetic_data.dropna()

        try:
            statistic, _ = ks_2samp(real_data, synthetic_data)
            return 1 - statistic
        except ValueError as e:
            if "Data passed to ks_2samp must not be empty" in str(e):
                return np.nan
            else:
                raise

    def compute_statistic(self, keys: pd.Series, values: pd.Series, statistic: str) -> pd.Series:
        """
        Compute a specified statistic for grouped data.

        Args:
            keys (pd.Series): Grouping keys (e.g., sequence IDs).
            values (pd.Series): Values to compute statistics on.
            statistic (str): Statistic to compute ('mean', 'median', 'std', 'length', 'inter_row_dep').

        Returns:
            pd.Series: Computed statistics grouped by keys.
        """
        df = pd.DataFrame({'keys': keys, 'values': values})

        if statistic == 'length':
            return df.groupby('keys').size()
        elif statistic == 'inter_row_dep':
            diffs = df['values'].shift(-self.x_step) - df['values']
            return diffs.groupby(df['keys']).mean()
        else:
            return df.groupby('keys')['values'].agg(statistic)

    def msas(self, real_data: tuple, synthetic_data: tuple, statistic: str) -> float:
        """
        Compute the MSAS metric for a given statistic.

        Args:
            real_data (tuple): Tuple of (keys, values) for real data.
            synthetic_data (tuple): Tuple of (keys, values) for synthetic data.
            statistic (str): Statistic to compute ('mean', 'median', 'std', 'length', 'inter_row_dep').

        Returns:
            float: MSAS score.
        """
        real_keys, real_values = real_data
        synthetic_keys, synthetic_values = synthetic_data

        real_stats = self.compute_statistic(real_keys, real_values, statistic)
        synthetic_stats = self.compute_statistic(synthetic_keys, synthetic_values, statistic)

        return self.ks(real_stats, synthetic_stats)

    def compute(self):
        """
        Compute MSAS scores for synthetic data and save the results to a CSV file.
        """
        print("[*] Starting MSAS computation...")
        results = []

        for synth_file in filter(lambda f: f.endswith('.csv'), os.listdir(self.synth_dir)):
            synth_path = os.path.join(self.synth_dir, synth_file)
            print(f"[+] Processing {synth_file}...")

            synth_data = pd.read_csv(synth_path)
            average_scores = {}

            for stat in self.statistics:
                scores = []
                for column in self.real_data.columns:
                    if column in self.exclude_columns:
                        continue

                    try:
                        score = self.msas(
                            real_data=(self.real_data['SID'], self.real_data[column]),
                            synthetic_data=(synth_data['SID'], synth_data[column]),
                            statistic=stat
                        )
                        scores.append(score)
                    except Exception as e:
                        print(f"[-] Error processing {column} with {stat}: {e}")
                        scores.append(np.nan)

                average_scores[stat] = np.nanmean(scores)

            overall_average = np.nanmean(list(average_scores.values()))
            epoch = int(synth_file.split('.')[0])
            results.append({'Epochs': epoch, 'MSAS': overall_average})

        results_df = pd.DataFrame(results).sort_values(by="Epochs").reset_index(drop=True)
        output_file = os.path.join(self.output_folder, "MSAS.csv")
        results_df.to_csv(output_file, index=False)
        print(f"[+] MSAS results saved to '{output_file}'.")

    def compute_all(self):
        """
        Execute the full MSAS process.
        """
        self.compute()