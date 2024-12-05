import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
import os

class Classifier:
    def __init__(self, param_grid=None, test_size=0.25, random_state=0):
        """
        Initialize the classifier with the given parameter grid, test size, and random state.

        :param param_grid: Dictionary of hyperparameters for GridSearchCV
        :param test_size: Proportion of data to be used as test set
        :param random_state: Seed used by the random number generator for data splitting
        """
        # Set the parameter grid for tuning the classifier if not provided
        if param_grid is None:
            param_grid = {
                'n_neighbors': [1, 3, 5, 7],
                'weights': ['uniform', 'distance'],
                'distance': ['euclidean', 'squared', 'manhattan']
            }
        self.param_grid = param_grid
        self.test_size = test_size
        self.random_state = random_state
        self.best_classifier = None  # Placeholder for the best classifier

    def load_and_process_data(self, file_path):
        """
        Load and process the dataset, grouping by 'SID' and extracting the required features and labels.

        :param file_path: Path to the CSV file containing the dataset
        :return: Tuple of features (X) and labels (y)
        """
        try:
            df = pd.read_csv(file_path)
            
            grouped = df.groupby('SID')
            
            X = []
            y = []

            for sid, group in grouped:
                if len(group) >= 96:  # Ensure there are enough data points
                    case = group.iloc[:96, 2:-1].values  # Select features as n_channels
                    label = group['Load_Type'].mode()[0]  # Majority label
                    X.append(case)
                    y.append(label)

            X = np.array(X)  # n_cases, n_channels, n_timepoints
            y = np.array(y)  # n_cases
            return X, y
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except pd.errors.EmptyDataError:
            print(f"Error: The file {file_path} is empty.")
        except Exception as e:
            print(f"An unexpected error occurred while loading the data: {e}")

    def tuning_classifier(self, X, y):
        """
        Tune the KNeighborsTimeSeriesClassifier using GridSearchCV and return the F1-score.

        :param X: Feature matrix
        :param y: Label vector
        :return: F1-score for the best classifier found by GridSearchCV
        """
        try:
            # Split the dataset into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            
            # Define the KNeighborsTimeSeriesClassifier
            classifier = KNeighborsTimeSeriesClassifier()
            
            # Set up GridSearchCV with the specified parameter grid
            grid_search = GridSearchCV(estimator=classifier, 
                                       param_grid=self.param_grid, 
                                       scoring='f1_macro', 
                                       cv=3, 
                                       n_jobs=-1, 
                                       verbose=1)
            
            # Fit GridSearchCV to the dataset
            grid_search.fit(X_train, y_train)
            
            # Get the best model from GridSearchCV
            self.best_classifier = grid_search.best_estimator_
            
            # Predict using the best classifier
            y_pred = self.best_classifier.predict(X_test)
            
            # Compute the F1-score for the test set
            f1 = f1_score(y_test, y_pred, average='macro')
            return f1
        except ValueError as e:
            print(f"Error: There was a value error during model tuning: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during model tuning: {e}")
    
    def evaluate_synthetic_datasets(self, real_f1, synthetic_folder_path):
        """
        Evaluate the synthetic datasets and compute the F1-ratio compared to the real dataset.

        :param real_f1: The F1-score from the real dataset
        :param synthetic_folder_path: Path to the folder containing synthetic datasets
        :return: DataFrame with the F1-ratios of synthetic datasets compared to the real dataset
        """
        f1_ratios = []

        # Get all synthetic dataset filenames
        synthetic_files = [f for f in os.listdir(synthetic_folder_path) if f.endswith('.csv')]

        # Loop through synthetic datasets
        for synthetic_file in synthetic_files:
            try:
                print(f"Processing synthetic dataset: {synthetic_file}")
                
                # Load and process the synthetic dataset
                X_synthetic, y_synthetic = self.load_and_process_data(os.path.join(synthetic_folder_path, synthetic_file))
                
                # Tune the classifier on the synthetic dataset and get F1-score
                f1_synthetic = self.tuning_classifier(X_synthetic, y_synthetic)
                
                # Compute the F1-ratio (real F1 / synthetic F1)
                f1_ratio = real_f1 / f1_synthetic

                # Extract epochs from the filename (remove '.csv' and convert to integer)
                epochs = int(synthetic_file.replace('.csv', ''))  # Remove .csv and convert to int
                
                # Append the results
                f1_ratios.append([epochs, f1_ratio])
            
            except Exception as e:
                print(f"Error processing {synthetic_file}: {e}")
        
        # Convert the results to a DataFrame
        f1_ratios_df = pd.DataFrame(f1_ratios, columns=['Epochs', 'F1-ratio'])
        return f1_ratios_df



# Main execution
if __name__ == "__main__":
    try:
        # Real dataset path
        real_dataset_path = './data/preprocessed_dataset.csv'
        classifier = Classifier()
        
        # Load and process the real dataset
        X_real, y_real = classifier.load_and_process_data(real_dataset_path)
        
        # Tune the classifier on the real dataset and get the F1-score
        real_f1 = classifier.tuning_classifier(X_real, y_real)
        
        # Folder with synthetic datasets
        synthetic_folder_path = './outputs/Synth_Data/'

        # TODO: print out the best classifiers parameters
        # Evaluate synthetic datasets and get the F1-ratio DataFrame
        f1_ratios_df = classifier.evaluate_synthetic_datasets(real_f1, synthetic_folder_path)

        # Sort the DataFrame by the 'Epochs' column
        f1_ratios_df = f1_ratios_df.sort_values(by='Epochs')

        # TODO: maybe index by epochs
        # Save the output to a CSV file
        f1_ratios_df.to_csv('f1_ratios.csv', index=False)
        print("Results saved to f1_ratios.csv")
    
    except Exception as e:
        print(f"An unexpected error occurred in the main execution: {e}")
