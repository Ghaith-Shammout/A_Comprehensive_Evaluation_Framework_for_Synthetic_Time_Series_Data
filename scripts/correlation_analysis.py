import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

class CorrelationAnalysis:
    """
    A class for performing Spearman correlation analysis between F1-ratios and evaluation metric values, 
    and plotting their relationship.
    """

    def __init__(self, f1_ratios_df, metric_file_path):
        """
        Initialize the CorrelationAnalysis class with the necessary data.

        :param f1_ratios_df: DataFrame containing the F1-ratios and Epochs of synthetic datasets.
        :param metric_file_path: Path to the CSV file containing the metric values and Epochs.
        """
        self.f1_ratios_df = f1_ratios_df
        self.metric_file_path = metric_file_path

    def correlation_analysis(self, metric_column):
        """
        Perform Spearman correlation analysis between F1-ratio and the specified metric values.

        :param metric_column: Column name of the metric to analyze (e.g., "AWD").
        :return: Spearman correlation coefficient and p-value, along with the merged DataFrame.
        """
        try:
            # Load the metric CSV file
            metric_df = pd.read_csv(self.metric_file_path)

            # Sort the metric DataFrame by 'Epochs'
            metric_df = metric_df.sort_values(by='Epochs')

            # Merge the F1-ratios DataFrame with the metric DataFrame on 'Epochs'
            merged_df = pd.merge(self.f1_ratios_df, metric_df[['Epochs', metric_column]], on='Epochs', how='inner')

            # Perform the Spearman correlation test between 'F1-ratio' and the specified metric column
            correlation, p_value = spearmanr(merged_df['F1-ratio'], merged_df[metric_column])

            # Print the Spearman correlation result
            print(f"Spearman correlation: {correlation}, p-value: {p_value}")

            # Return the correlation, p-value, and merged DataFrame
            return correlation, p_value, merged_df

        except Exception as e:
            print(f"Error in correlation analysis: {e}")
            return None, None, None

    def plot_correlation(self, X, Y, output_path, metric_name):
        """
        Plot the scatter plot of two variables and display the Spearman correlation.

        :param X: First variable (array-like, e.g., 'F1-ratio').
        :param Y: Second variable (array-like, e.g., the metric values).
        :param output_path: Path to save the correlation plot.
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

            # Save the plot to a file
            plt.savefig(output_path, bbox_inches='tight')  # Save as PNG
            print(f"Plot saved as '{output_path}'")
            plt.close()  # Close the plot to free memory
        
        except Exception as e:
            print(f"Error in plotting correlation: {e}")

    
# Main execution
if __name__ == "__main__":
    try:
        """
        Main function to execute correlation analysis and plotting.
        """
        f1_ratios_df = pd.read_csv('./outputs/Evaluation/f1_ratios.csv')
    
        # Example MSAS file path (replace with actual file path)
        msas_file_path = './outputs/Evaluation/MSAS.csv'

        metric_column = 'MSAS'
        
        # Initialize the CorrelationAnalysis object
        analysis = CorrelationAnalysis(f1_ratios_df, msas_file_path)
    
        # Perform correlation analysis
        correlation, p_value, merged_df = analysis.correlation_analysis(metric_column)
    
        # If the correlation analysis was successful, plot the correlation
        if correlation is not None and p_value is not None:
            analysis.plot_correlation(merged_df['F1-ratio'], merged_df[metric_column], 'plot', metric_column)
        else:
            print("Correlation analysis failed. No plot will be generated.")




    except Exception as e:
        print(f"An unexpected error occurred in the main execution: {e}")
