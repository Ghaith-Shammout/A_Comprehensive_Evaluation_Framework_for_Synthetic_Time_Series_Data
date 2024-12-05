import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

class CorrelationAnalysis:
    """
    A class for performing Spearman correlation analysis between F1-ratios and MSAS values, 
    and plotting their relationship.
    """

    def __init__(self, f1_ratios_df, msas_file_path):
        """
        Initialize the CorrelationAnalysis class with the necessary data.

        :param f1_ratios_df: DataFrame containing the F1-ratios and Epochs of synthetic datasets
        :param msas_file_path: Path to the CSV file containing the 'MSAS' column
        """
        self.f1_ratios_df = f1_ratios_df
        self.msas_file_path = msas_file_path

    def correlation_analysis(self):
        """
        Perform Spearman correlation analysis between F1-ratio and MSAS values.

        :return: Spearman correlation coefficient and p-value, along with the merged DataFrame
        """
        try:
            # Load the MSAS CSV file
            msas_df = pd.read_csv(self.msas_file_path)

            # Sort the MSAS dataframe by 'Epochs'
            msas_df = msas_df.sort_values(by='Epochs')

            # Merge the F1-ratios DataFrame with the MSAS DataFrame on 'Epochs'
            merged_df = pd.merge(self.f1_ratios_df, msas_df[['Epochs', 'MSAS']], on='Epochs', how='inner')

            # Perform the Spearman correlation test between 'F1-ratio' and 'MSAS'
            correlation, p_value = spearmanr(merged_df['F1-ratio'], merged_df['MSAS'])

            # Print the Spearman correlation result
            print(f"Spearman correlation: {correlation}, p-value: {p_value}")

            # Return the correlation, p-value, and merged dataframe
            return correlation, p_value, merged_df

        except Exception as e:
            print(f"Error in correlation analysis: {e}")
            return None, None, None

    def plot_correlation(self, X, Y):
        """
        Plot the scatter plot of two variables and display the Spearman correlation.

        :param X: First variable (array-like)
        :param Y: Second variable (array-like)
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
            plt.ylabel('MSAS', fontsize=12)
            plt.title('Correlation between F1-ratio and MSAS', fontsize=14)

            # Save the plot to a file
            plt.savefig("MSAS_correlation_plot.png", bbox_inches='tight')  # Save as PNG
            print("Plot saved as 'MSAS_correlation_plot.png'")
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
    
        # Initialize the CorrelationAnalysis object
        analysis = CorrelationAnalysis(f1_ratios_df, msas_file_path)
    
        # Perform correlation analysis
        correlation, p_value, merged_df = analysis.correlation_analysis()
    
        # If the correlation analysis was successful, plot the correlation
        if correlation is not None and p_value is not None:
            analysis.plot_correlation(merged_df['F1-ratio'], merged_df['MSAS'])
        else:
            print("Correlation analysis failed. No plot will be generated.")




    except Exception as e:
        print(f"An unexpected error occurred in the main execution: {e}")
