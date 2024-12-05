import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score
import os

class Classifier:
    def __init__(self):
        """
        Initialize the classifier with the given parameter grid, test size, and random state.
        """
        pass  # No additional initialization needed for now
    
    def load_and_process_data(self, file_path, seq_index_col='SID', target_col=None):
        """
        Load and process the dataset, grouping by 'SID' and extracting the required features and labels.

        :param file_path: Path to the CSV file containing the dataset
        :param seq_index_col: Column name used for grouping sequences (default is 'SID')
        :param target_col: Column name for the target label
        :return: Tuple of features (X) and labels (y)
        """
        try:
            # Read the dataset from CSV
            df = pd.read_csv(file_path)

            # Group by the 'seq_index' column
            grouped = df.groupby(seq_index_col)
            
            # Initialize lists for features (X) and labels (y)
            X = []
            y = []
            
            # Iterate through each group
            for _, group in grouped:
                # Drop the sequence index column ('seq_index_col') after grouping
                group = group.drop(columns=[seq_index_col])
        
                # Slice the data: select all columns except the last one (target label column)
                features = group.iloc[:, 1:-1].values  # All columns except the last column
                label = group[target_col].mode()[0]  # Apply majority vote to the target label
                
                # Append the features and label to the lists
                X.append(features)
                y.append(label)
            
            # Convert the lists to numpy arrays
            X = np.array(X)  # Features: (n_cases, n_features)
            y = np.array(y)  # Labels: (n_cases,)
            
            return X, y
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return None, None  # Explicitly return None for both X and y
        except pd.errors.EmptyDataError:
            print(f"Error: The file {file_path} is empty.")
            return None, None  # Explicitly return None for both X and y
        except Exception as e:
            print(f"An unexpected error occurred while loading the data: {e}")
            return None, None  # Explicitly return None for both X and y
            
    def tuning_classifier(self, X, y, param_grid, test_size, random_state, metric='f1'):
        """
        Tune the KNeighborsTimeSeriesClassifier using GridSearchCV and return the specified evaluation metric score.
    
        :param X: Feature matrix
        :param y: Label vector
        :param metric: The evaluation metric to use. Options are 'accuracy', 'f1', 'precision'. Default is 'f1'.
        :return: The score for the best classifier found by GridSearchCV based on the specified metric.
        """
        try:
            # Split the dataset into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=test_size,
                                                                random_state=random_state)
            
            # Define the KNeighborsTimeSeriesClassifier
            classifier = KNeighborsTimeSeriesClassifier()
            
            # Determine the scoring method based on the provided metric
            if metric == 'accuracy':
                scoring = 'accuracy'
                score_function = accuracy_score
            elif metric == 'f1':
                scoring = 'f1_macro'
                score_function = f1_score
            elif metric == 'precision':
                scoring = 'precision_macro'
                score_function = precision_score
            else:
                raise ValueError(f"Unknown metric: {metric}. Choose from 'accuracy', 'f1', or 'precision'.")
            
            # Set up GridSearchCV with the specified parameter grid and metric
            grid_search = GridSearchCV(estimator=classifier, 
                                       param_grid=param_grid, 
                                       scoring=scoring, 
                                       cv=3, 
                                       n_jobs=-1, 
                                       verbose=1)
            
            # Fit GridSearchCV to the dataset
            grid_search.fit(X_train, y_train)
            
            # Get the best model from GridSearchCV
            self.best_classifier = grid_search.best_estimator_
            
            # Predict using the best classifier
            y_pred = self.best_classifier.predict(X_test)
            
            # Compute the evaluation metric for the test set
            score = score_function(y_test, y_pred, average='macro')
            
            return score
        except ValueError as e:
            print(f"Error: There was a value error during model tuning: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during model tuning: {e}")
            return None
    
    def classify(self, real_dataset_path, synthetic_folder_path, 
                 seq_index_col, target_col, metric, param_grid,
                 test_size, random_state):
        """
        Evaluate the synthetic datasets and compute the F1-ratio compared to the real dataset.

        :param real_dataset_path: Path to the real dataset
        :param synthetic_folder_path: Path to the folder containing synthetic datasets
        :return: DataFrame with the F1-ratios of synthetic datasets compared to the real dataset
        """
        # Load and process the real dataset
        X_real, y_real = self.load_and_process_data(real_dataset_path, seq_index_col, target_col)
        
        # If loading real dataset failed, return an empty DataFrame
        if X_real is None or y_real is None:
            print("Real dataset could not be loaded.")
            return pd.DataFrame()

        # Tune the classifier on the real dataset and get the F1-score
        real_f1 = self.tuning_classifier(X_real, y_real, param_grid, test_size, random_state, metric)
        
        f1_ratios = []

        # Get all synthetic dataset filenames
        synthetic_files = [f for f in os.listdir(synthetic_folder_path) if f.endswith('.csv')]

        # Loop through synthetic datasets
        for synthetic_file in synthetic_files:
            try:
                print(f"Processing synthetic dataset: {synthetic_file}")
                
                # Load and process the synthetic dataset
                X_synthetic, y_synthetic = self.load_and_process_data(os.path.join(synthetic_folder_path, synthetic_file),
                                                                      seq_index_col,
                                                                      target_col)
                
                # If loading synthetic dataset failed, skip it
                if X_synthetic is None or y_synthetic is None:
                    print(f"Skipping synthetic dataset {synthetic_file} due to loading failure.")
                    continue

                # Tune the classifier on the synthetic dataset and get F1-score
                f1_synthetic = self.tuning_classifier(X_synthetic, y_synthetic, param_grid, test_size, random_state, metric)
                
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

        # Sort the DataFrame by the 'Epochs' column
        f1_ratios_df = f1_ratios_df.sort_values(by='Epochs')

        # Save the output to a CSV file
        f1_ratios_df.to_csv('./outputs/Evaluation/f1_ratios.csv', index=False)
        print("Results saved to f1_ratios.csv")
        
        return f1_ratios_df


# Main execution
if __name__ == "__main__":
    try:
        # Real dataset path
        real_dataset_path = './data/preprocessed_dataset.csv'
        classifier = Classifier()
        
        # Load and process the real dataset
        X_real, y_real = classifier.load_and_process_data(real_dataset_path, seq_index_col, target_col)
        
        # Tune the classifier on the real dataset and get the F1-score
        real_f1 = classifier.tuning_classifier(X_real, y_real, metric)
        
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
