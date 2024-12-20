import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

class MSAS:
    def __init__(self, real_file, synth_dir, output_folder, x_step, exclude_columns):
        """
        Initialize the MSAS class.

        Args:
            real_file (str): Path to the real data file.
            synth_dir (str): Path to the directory containing synthetic data files.
            output_folder (str): Path to save the results CSV file.
            x_step (int): Step size for inter-row dependency calculation (default is 1).
        """
        self.real_file = real_file
        self.synth_dir = synth_dir
        self.output_folder = output_folder
        self.x_step = x_step
        self.exclude_columns = exclude_columns
        self.real_data = pd.read_csv(real_file)
        self.statistics = ['mean', 'median', 'std', 'length', 'inter_row_dep']

    def ks(self, real_data, synthetic_data):
        """Compare two continuous columns using a Kolmogorov–Smirnov test."""
        real_data = pd.Series(real_data).dropna()
        synthetic_data = pd.Series(synthetic_data).dropna()

        try:
            statistic, _ = ks_2samp(real_data, synthetic_data)
        except ValueError as e:
            if str(e) == '[-] Data passed to ks_2samp must not be empty':
                return np.nan
            else:
                raise ValueError(e)

        return 1 - statistic

    def msas(self, real_data, synthetic_data, statistic='mean'):
        """Compute the MSAS metric."""
        valid_statistics = ['mean', 'median', 'std', 'length', 'inter_row_dep']
        if statistic not in valid_statistics:
            raise ValueError(f'[-] Invalid statistic: {statistic}. Choose from {valid_statistics}.')

        for data in [real_data, synthetic_data]:
            if not isinstance(data, tuple) or len(data) != 2 or not (isinstance(data[0], pd.Series) and isinstance(data[1], pd.Series)):
                raise ValueError('The data must be a tuple of two pandas series.')

        real_keys, real_values = real_data
        synthetic_keys, synthetic_values = synthetic_data

        def calculate_statistics(keys, values):
            df = pd.DataFrame({'keys': keys, 'values': values})

            if statistic == 'length':
                return df.groupby('keys').size()  # Return the length of each sequence
            elif statistic == 'inter_row_dep':
                # Calculate inter-row dependencies (average difference for x steps ahead)
                diffs = df['values'].shift(-self.x_step) - df['values']
                return diffs.groupby(df['keys']).mean()  # Return average difference for each sequence
            else:
                return df.groupby('keys')['values'].agg(statistic)  # Calculate the other statistics

        # Calculate statistics for real and synthetic data
        real_stats = calculate_statistics(real_keys, real_values)
        synthetic_stats = calculate_statistics(synthetic_keys, synthetic_values)

        return self.ks(real_stats, synthetic_stats)

    def compute(self):
        """Compute MSAS scores for synthetic data and save results."""
        # Initialize an empty list to store results for each synthetic file
        results = []

        # Loop through each CSV file in the directory
        for synth_file in os.listdir(self.synth_dir):
            if synth_file.endswith(".csv"):
                synth_path = os.path.join(self.synth_dir, synth_file)
                print(f"[+] Processing synthetic file: {synth_file}")

                # Read the synthetic data
                synth = pd.read_csv(synth_path)

                # Initialize a dictionary to store average scores for each statistic
                average_scores = {}

                # Loop through each statistic and calculate the score for each column
                for stat in self.statistics:
                    # Initialize a list to store the scores for the current statistic
                    scores = []

                    #exclude_columns = ['SID', 'date', 'Load_Type']
                    # Loop through each column in the 'real' DataFrame (excluding specific columns)
                    for column in self.real_data.columns:
                        if column in self.exclude_columns: continue

                        try:
                            # Compute MSAS score for this column and statistic
                            score = self.msas(
                                real_data=(self.real_data['SID'], self.real_data[column]),
                                synthetic_data=(synth['SID'], synth[column]),
                                statistic=stat  # Using the current statistic
                            )
                            # Store the computed score in the list
                            scores.append(score)
                        except Exception as e:
                            print(f"[-] Error processing column {column}: {e}")
                            scores.append(np.nan)  # If there's an error, append NaN to the scores

                    # Calculate the average score for the current statistic
                    average_score = np.nanmean(scores)
                    average_scores[stat] = average_score
                    #print(f"Computed {stat} score across all columns: {average_score}")

                # Now, calculate the overall average of all statistics
                overall_average = np.nanmean(list(average_scores.values()))

                # Extract the epoch number from the file name (e.g., 200.csv -> 200)
                epoch = int(synth_file.split('.')[0])

                # Append the results (epoch number and overall average) to the list
                results.append({'Epochs': epoch, 'MSAS': overall_average})

                #print(f"[+] Overall average score for {synth_file}: {overall_average}")

        # Convert the results list into a DataFrame
        results_df = pd.DataFrame(results)

        # Step 9: Convert the 'Epochs' column to numeric, coercing errors (invalid values become NaN)
        results_df['Epochs'] = pd.to_numeric(results_df['Epochs'], errors='coerce')
    
        # Step 10: Sort the DataFrame by 'Epochs' and reset the index
        results_df = results_df.sort_values(by="Epochs").reset_index(drop=True)

        # Save the results to the output CSV file
        results_df.to_csv(f"{self.output_folder}/MSAS.csv", index=False)

        print(f"[+] MSAS Calculation Completed & Results Saved to {self.output_folder}/msas.csv")

# Usage example
if __name__ == "__main__":
    synth_directory = './Synth'  # Replace with the path to your directory containing synthetic CSVs
    real_data_file = './real_data.csv'   # Path to your real data file
    output_csv_file = './MSAS.csv' # Path to save the results

    # Create an MSAS object and compute scores
    msas_calculator = MSAS(real_file=real_data_file, synth_dir=synth_directory, output_folder=output_csv_file)
    msas_calculator.compute()
