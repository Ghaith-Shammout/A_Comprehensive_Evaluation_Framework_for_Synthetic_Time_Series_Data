import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
import logging

class CorrelationAnalysis:
    """
    A class for performing Spearman correlation analysis between F1-ratios and evaluation metric values, 
    and plotting their relationship.
    """

    def __init__(self):
        """
        Initialize the CorrelationAnalysis class with the necessary data.
        """
        pass

    def correlation_analysis(self, f1_ratios_path, metric_columns, base_name, output_dir, correlation_output_path):
        """
        Perform correlation analysis for each metric in the list, save results to CSV, and generate plots.
        :param f1_ratios_path: Path to F1-ratio file.
        :param metric_columns: List of metric columns to analyze.
        :param output_dir: Directory where correlation plots will be saved.
        :param correlation_output_path: Path to save the CSV file with correlation results.
        """
        print("[+] Correlation Analysis Started")
        correlation_results = []  # List to store the results for each metric

        for metric_column in metric_columns:
            # Get the file path for the current metric
            metric_path = f"./outputs/{base_name}/Evaluation/{metric_column}.csv"
            

            if not os.path.exists(metric_path):
                logging.warning(f"Metric file path for {metric_column} not found. Skipping.")
                continue  # Skip to the next metric if path is missing

            print(f"[+] Processing {metric_path}")
            logging.info(f"Processing correlation for {metric_column}...")

            # Read the specific metric file and perform correlation analysis
            correlation, p_value, merged_df = self.calc_correlation(f1_ratios_path, metric_column, metric_path)

            if correlation is not None and p_value is not None:
                # Store results for later saving to CSV
                correlation_results.append({
                    'metric': metric_column,
                    'correlation': correlation,
                    'p_value': p_value
                })

                # Plot the correlation
                self.plot_correlation(
                    X=merged_df['F1-ratio'], 
                    Y=merged_df[metric_column],
                    output_dir=output_dir,
                    metric_name=metric_column
                )
            else:
                logging.warning(f"Correlation analysis for {metric_column} failed. Skipping plot generation.")

        # Save the correlation results to CSV
        if correlation_results:
            df = pd.DataFrame(correlation_results)
            df.to_csv(f"{correlation_output_path}/correlation_results.csv", index=False)
            logging.info(f"Correlation results saved to {correlation_output_path}")
        else:
            logging.warning("No valid correlation results to save.")

    def calc_correlation(self, f1_ratios_path, metric_column, metric_file_path):
        """
        Perform Spearman correlation analysis between F1-ratio and the specified metric values.
        :param f1_ratios_path: Path to the F1-ratios CSV.
        :param metric_column: Metric column name (e.g., "AWD").
        :param metric_file_path: Path to the metric file.
        :return: Spearman correlation coefficient and p-value, along with the merged DataFrame.
        """
        try:
            f1_ratios_df = pd.read_csv(f1_ratios_path)
            metric_df = pd.read_csv(metric_file_path)

            # Sort the metric DataFrame by 'Epochs' for proper alignment
            metric_df = metric_df.sort_values(by='Epochs')

            # Merge the DataFrames on 'Epochs'
            merged_df = pd.merge(f1_ratios_df, metric_df[['Epochs', metric_column]], on='Epochs', how='inner')

            # Ensure data is numeric and handle missing values
            merged_df.dropna(subset=['F1-ratio', metric_column], inplace=True)
            merged_df = merged_df[pd.to_numeric(merged_df['F1-ratio'], errors='coerce').notnull()]
            merged_df = merged_df[pd.to_numeric(merged_df[metric_column], errors='coerce').notnull()]

            # Perform Spearman correlation test
            correlation, p_value = spearmanr(merged_df['F1-ratio'], merged_df[metric_column])

            print(f"Spearman correlation for {metric_column}: {correlation:.3f}, p-value: {p_value:.3f}")
            logging.info(f"Spearman correlation for {metric_column}: {correlation:.3f}, p-value: {p_value:.3f}")

            return correlation, p_value, merged_df
        except Exception as e:
            logging.error(f"Error in correlation analysis for {metric_column}: {e}")
            return None, None, None

    def plot_correlation(self, X, Y, output_dir, metric_name):
        """
        Plot the scatter plot of two variables and display the Spearman correlation.
        :param X: First variable (array-like, e.g., 'F1-ratio').
        :param Y: Second variable (array-like, e.g., the metric values).
        :param output_dir: Path to save the correlation plot.
        :param metric_name: Name of the metric being analyzed (e.g., "AWD").
        """
        try:
            corr, _ = spearmanr(X, Y)
            
            # Create scatter plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=X, y=Y, color='blue', label="Data points")
            plt.text(0.1, 0.9, f'Spearman Correlation: {corr:.2f}', 
                     transform=plt.gca().transAxes, fontsize=14, color='red')

            # Optional: Fit a regression line (if you'd like a trendline)
            sns.regplot(x=X, y=Y, scatter=False, color='green', line_kws={'lw': 2, 'ls': '--'})

            # Add labels and title
            plt.xlabel('F1-ratio', fontsize=12)
            plt.ylabel(metric_name, fontsize=12)
            plt.title(f'Correlation between F1-ratio and {metric_name}', fontsize=14)

            # Customize the output file path
            output_path = os.path.join(output_dir, f"{metric_name}.png")
            plt.savefig(output_path, bbox_inches='tight')  # Save as PNG
            logging.info(f"Plot for {metric_name} saved as '{output_path}'")
            plt.close()  # Close the plot to free memory
        except Exception as e:
            logging.error(f"Error in plotting correlation for {metric_name}: {e}")
