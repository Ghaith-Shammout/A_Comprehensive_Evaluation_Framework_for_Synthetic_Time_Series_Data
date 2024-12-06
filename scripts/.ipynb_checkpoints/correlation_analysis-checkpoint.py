# Updated CorrelationAnalysis class
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

class CorrelationAnalysis:
    """
    A class for performing Spearman correlation analysis between F1-ratios and evaluation metric values, 
    and plotting their relationship.
    """

    def __init__(self):
        """
        Initialize the CorrelationAnalysis class with the necessary data.

        :param f1_ratios_df: Path to the CSV file containing F1-ratios and Epochs.
        """


    def correlation_analysis(self, f1_ratios_path, metric_columns, metric_file_paths, output_dir, correlation_output_path):
        """
        Perform correlation analysis for each metric in the list, save results to CSV, and generate plots.

        :param metric_columns: List of metric columns to analyze.
        :param metric_file_paths: Dictionary containing file paths for each metric.
        :param output_dir: Directory where correlation plots will be saved.
        :param correlation_output_path: Path to save the CSV file with correlation results.
        """
        
        correlation_results = []  # List to store the results for each metric

        # Loop through each metric and perform analysis
        for metric_column in metric_columns:
            # Get the file path for the current metric
            metric_path = metric_file_paths.get(metric_column)

            if not metric_path:
                print(f"Metric file path for {metric_column} not found. Skipping.")
                continue  # Skip to the next metric if path is missing

            # Read the specific metric file and perform correlation analysis for the current metric
            correlation, p_value, merged_df = self.calc_correlation(f1_ratios_path, metric_column, metric_path)

            if correlation is not None and p_value is not None:
                # Save the result (metric, correlation, p_value)
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
                print(f"Correlation analysis for {metric_column} failed. Skipping plot generation.")
        
        # Save the correlation results to a CSV file after processing all metrics
        if correlation_results:
            df = pd.DataFrame(correlation_results)
            df.to_csv(correlation_output_path, index=False)
            print(f"Correlation results saved to {correlation_output_path}")
        else:
            print("No valid correlation results to save.")

    def calc_correlation(self, f1_ratios_path, metric_column, metric_file_path):
        """
        Perform Spearman correlation analysis between F1-ratio and the specified metric values.

        :param metric_column: Column name of the metric to analyze (e.g., "AWD").
        :param metric_file_path: Path to the metric file for the specific column.
        :return: Spearman correlation coefficient and p-value, along with the merged DataFrame.
        """
        try:
            f1_ratios_df = pd.read_csv(f1_ratios_path)
            
            # Load the metric CSV file
            metric_df = pd.read_csv(metric_file_path)

            # Sort the metric DataFrame by 'Epochs'
            metric_df = metric_df.sort_values(by='Epochs')

            # Merge the F1-ratios DataFrame with the metric DataFrame on 'Epochs'
            merged_df = pd.merge(f1_ratios_df, metric_df[['Epochs', metric_column]], on='Epochs', how='inner')

            # Perform the Spearman correlation test between 'F1-ratio' and the specified metric column
            correlation, p_value = spearmanr(merged_df['F1-ratio'], merged_df[metric_column])

            # Print the Spearman correlation result
            print(f"Spearman correlation for {metric_column}: {correlation}, p-value: {p_value}")

            # Return the correlation, p-value, and merged DataFrame
            return correlation, p_value, merged_df

        except Exception as e:
            print(f"Error in correlation analysis for {metric_column}: {e}")
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
            # Calculate Spearman correlation
            corr, _ = spearmanr(X, Y)
            
            # Create scatter plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=X, y=Y, color='blue', label="Data points")

            # Add Spearman correlation to the plot
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
            
            # Save the plot to a file
            plt.savefig(output_path, bbox_inches='tight')  # Save as PNG
            print(f"Plot for {metric_name} saved as '{output_path}'")
            plt.close()  # Close the plot to free memory
        
        except Exception as e:
            print(f"Error in plotting correlation for {metric_name}: {e}")
