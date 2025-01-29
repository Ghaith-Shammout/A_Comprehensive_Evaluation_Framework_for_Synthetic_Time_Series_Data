import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path
import logging
from scipy.stats import norm  # For critical z-values

class CorrelationAnalysis:
    """
    A class for performing Spearman correlation analysis between F1-ratios and evaluation metric values,
    and plotting their relationship.
    """

    def __init__(self):
        """Initialize the CorrelationAnalysis class."""
        pass

    def calculate_averages_and_save(self, evaluation_path, folders):
        """
        Calculate the average of the second column in each CSV file within the specified folders,
        sort the results by 'Epochs', and save the results in a new CSV file named after each folder.

        :param evaluation_path: Base directory containing the folders (e.g., "./outputs/{source_name}/Evaluation").
        :param folders: List of folder names to process (e.g., ['MSAS', 'AWD', 'TC', 'f1']).
        """
        for folder in folders:
            folder_path = os.path.join(evaluation_path, folder)
            # Check if the folder exists
            if not os.path.exists(folder_path):
                print(f"[-] Folder {folder_path} does not exist. Skipping...")
                continue
            # Initialize a list to store the results
            results = []
            # Iterate through each CSV file in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        # Read the CSV file
                        df = pd.read_csv(file_path)
                        # Check if the CSV has at least two columns
                        if len(df.columns) < 2:
                            print(f"[-] File {file_name} in folder {folder} does not have enough columns. Skipping...")
                            continue
                        # Calculate the average of the second column
                        average_value = df.iloc[:, 1].mean()
                        # Extract the number from the file name (e.g., "100.csv" -> 100)
                        try:
                            file_name_number = int(file_name.split('.')[0])
                        except ValueError:
                            print(f"[-] File {file_name} in folder {folder} does not have a valid numeric prefix. Skipping...")
                            continue
                        # Append the result to the list
                        results.append({'Epochs': file_name_number, f'{folder}': average_value})
                    except Exception as e:
                        print(f"[-] Error processing file {file_name} in folder {folder}: {e}")
                        continue
            # Convert the results to a DataFrame
            results_df = pd.DataFrame(results)
            # Sort the DataFrame by 'Epochs' before saving
            results_df = results_df.sort_values(by='Epochs')
            # Save the results to a new CSV file named after the folder
            output_file_path = os.path.join(evaluation_path, f"{folder}.csv")
            try:
                results_df.to_csv(output_file_path, index=False)
                print(f"[+] Processed {folder} and saved results to {output_file_path}")
            except Exception as e:
                print(f"[-] Error saving results for folder {folder}: {e}")

    def compute_correlation(self, evaluation_path: str, metric_columns: list[str], plot_path: str):
        """
        Perform correlation analysis for each metric in the list, save results to CSV, and generate plots.

        Args:
            evaluation_path (str): Path to evaluation measures
            metric_columns (list[str]): List of metric columns to analyze.
            plot_path (str): Directory where correlation plots will be saved.
        """
        print("[+] Correlation Analysis Started")
        f1_ratios_df = pd.read_csv(f"{evaluation_path}/f1.csv")
        correlation_results = []  # To store results for each metric

        for metric_column in metric_columns:
            print(f"[+] Processing metric: {metric_column}")
            logging.info(f"Processing correlation for {metric_column}...")
            metric_df = pd.read_csv(f"{evaluation_path}/{metric_column}.csv")

            # Extract F1-ratios, metric values, and epoch numbers
            f1_ratios, metric_values, epochs = CorrelationAnalysis.extract_metric_values(f1_ratios_df, metric_df, metric_column)

            correlation, p_value = CorrelationAnalysis.spearman_correlation(f1_ratios, metric_values)
            logging.info(f"Spearman correlation for {metric_column}: {correlation:.3f}, p-value: {p_value:.3f}")

            if correlation is not None and p_value is not None:
                # Calculate confidence interval
                ci_lower, ci_upper = self.calculate_spearman_ci(correlation, len(f1_ratios), confidence_level=0.99)
                logging.info(f"95% Confidence Interval for {metric_column}: ({ci_lower:.3f}, {ci_upper:.3f})")

                correlation_results.append({
                    "metric": metric_column,
                    "correlation": correlation,
                    "p_value": p_value,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper
                })

                # Plot correlation with epoch annotations
                CorrelationAnalysis.plot_correlation(
                    X=f1_ratios,
                    Y=metric_values,
                    corr=correlation,
                    plot_path=plot_path,
                    metric_name=metric_column,
                    epochs=epochs  # Pass epoch numbers
                )
            else:
                logging.warning(f"Correlation analysis failed for {metric_column}. Skipping plot generation.")

        # Save correlation results
        CorrelationAnalysis.save_correlation_results(correlation_results, evaluation_path)

    @staticmethod
    def calculate_spearman_ci(correlation: float, n: int, confidence_level: float = 0.95):
        """
        Calculate the confidence interval for a Spearman correlation coefficient using Fisher's z-transformation.

        Args:
            correlation (float): Spearman correlation coefficient.
            n (int): Sample size.
            confidence_level (float): Confidence level for the interval (default: 0.95).

        Returns:
            tuple: Lower and upper bounds of the confidence interval.
        """
        if n <= 3:
            raise ValueError("Sample size must be greater than 3 to calculate confidence interval.")

        # Fisher's z-transformation
        z = np.arctanh(correlation)
        se_z = 1 / np.sqrt(n - 3)  # Standard error of z

        # Critical value for the confidence level
        z_critical = norm.ppf((1 + confidence_level) / 2)

        # Confidence interval for z
        z_lower = z - z_critical * se_z
        z_upper = z + z_critical * se_z

        # Transform back to correlation scale
        ci_lower = np.tanh(z_lower)
        ci_upper = np.tanh(z_upper)

        return ci_lower, ci_upper

    @staticmethod
    def save_correlation_results(correlation_results: list, evaluation_path: str):
        """
        This function attempts to save correlation results to a CSV file.

        Parameters:
        - correlation_results: List or dictionary of correlation data to be saved.
        - evaluation_path: Path where the CSV file will be saved.

        Logs a message if the results are successfully saved or if an error occurs.
        """
        try:
            results_df = pd.DataFrame(correlation_results)
            results_csv_path = f"{evaluation_path}/correlation.csv"
            results_df.to_csv(results_csv_path, index=False)
            logging.info(f"Correlation results saved to {results_csv_path}")
        except ValueError as ve:
            logging.warning(str(ve))
        except Exception as e:
            logging.error(f"An error occurred while saving the correlation results: {str(e)}")

    @staticmethod
    def extract_metric_values(f1_ratios_df, metric_df, metric_column):
        """
        Merges two DataFrames on 'Epochs' and filters out rows where 'F1-ratio'
        or the specified metric column have invalid (non-numeric) values.

        Parameters:
        f1_ratios_df (pd.DataFrame): DataFrame containing 'Epochs' and 'F1-ratio'.
        metric_df (pd.DataFrame): DataFrame containing 'Epochs' and the metric column.
        metric_column (str): The name of the column in metric_df to merge with 'F1-ratio'.

        Returns:
        tuple: Three variables - one for valid 'F1-ratio' values, one for valid metric values, and one for epoch numbers.
        """
        # Step 1: Merge the two DataFrames on 'Epochs'
        merged_df = pd.merge(
            f1_ratios_df, metric_df[["Epochs", metric_column]], on="Epochs", how="inner"
        )

        # Step 3: Extract valid 'F1-ratio', metric values, and epoch numbers
        f1_ratios = merged_df["f1"].values
        metric_values = merged_df[metric_column].values
        epochs = merged_df["Epochs"].values

        # Return the three variables as a tuple
        return f1_ratios, metric_values, epochs

    @staticmethod
    def spearman_correlation(f1_ratios: np.ndarray, metric_values: np.ndarray):
        """
        Perform Spearman correlation analysis between F1-ratio and the specified metric values.

        Args:
            f1_ratios (numpy.ndarray): Path to the F1-ratios CSV.
            metric_values (numpy.ndarray): Metric column name (e.g., "AWD").

        Returns:
            tuple: Spearman correlation coefficient, p-value, and merged DataFrame.
        """
        try:
            correlation, p_value = spearmanr(f1_ratios, metric_values)
            return correlation, p_value
        except Exception as e:
            logging.error(f"Error calculating correlation for: {e}")
            return None, None

    @staticmethod
    def plot_correlation(X: np.ndarray, Y: np.ndarray, corr: float, plot_path: str, metric_name: str, epochs: np.ndarray):
        """
        Plot the scatter plot of two variables and display the Spearman correlation.
        Annotate each data point with its corresponding epoch number.

        Args:
            X (np.ndarray): First variable (e.g., "F1-ratio").
            Y (np.ndarray): Second variable (e.g., metric values).
            corr (float): Spearman correlation coefficient.
            plot_path (str): Directory to save the correlation plot.
            metric_name (str): Name of the metric being analyzed.
            epochs (np.ndarray): Array of epoch numbers corresponding to the data points.
        """
        try:
            plt.figure(figsize=(10, 8))  # Slightly larger figure for better readability
            sns.scatterplot(x=X, y=Y, color="blue", label="Data points", s=60, marker='o')
            sns.regplot(x=X, y=Y, scatter=False, color="green", line_kws={"lw": 3, "ls": "--"}, ci=None)

            # Annotate each point with its epoch number
            for i, epoch in enumerate(epochs):
                plt.text(X[i], Y[i], f'E{epoch}', fontsize=9, ha='right', va='bottom', color='black')

            # Display Spearman correlation
            plt.text(0.05, 0.85, f"Spearman Correlation: {corr:.2f}", transform=plt.gca().transAxes,
                     fontsize=14, color="red", bbox=dict(facecolor='white', alpha=0.5))

            plt.xlabel("F1-ratio", fontsize=14)
            plt.ylabel(f"{metric_name} (score)", fontsize=14)
            plt.grid(visible=True, linestyle="--", alpha=0.7)
            plt.title(f"Correlation between F1-ratio and {metric_name}", fontsize=16, fontweight="bold")

            output_path = f"{plot_path}/{metric_name}.png"
            plt.savefig(output_path, bbox_inches="tight")
            logging.info(f"Plot saved to {output_path}")
        except Exception as e:
            logging.error(f"Error plotting correlation for {metric_name}: {e}")