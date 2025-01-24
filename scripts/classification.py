import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier


class Classifier:
    """
    A class for evaluating real and synthetic datasets using time-series classification 
    and computing F1-ratios between real and synthetic datasets.
    """

    def __init__(self):
        """Initialize the Classifier class."""
        pass
        
    def compute(self, real_dataset_path: str, synthetic_folder_path: str, seq_index_col: str, target_col: str,
                 param_grid: dict, test_size: float, random_state: int, output_dir: str):
        """
        Compute real and synthetic datasets and compute F1-ratios.

        Args:
            real_dataset_path (str): Path to the real dataset.
            synthetic_folder_path (str): Path to the folder containing synthetic datasets.
            seq_index_col (str): Sequence index column.
            target_col (str): Target label column.
            param_grid (dict): Hyperparameter grid for tuning the classifier.
            test_size (float): Test set proportion.
            random_state (int): Random state for reproducibility.
            output_dir (str): Directory to save the F1-ratio results.

        Returns:
            pd.DataFrame: DataFrame containing F1-ratios.
        """
        print("[+] Starting classification process...")
        X_train_real, X_test_real, y_train_real, y_test_real = self.split_data(real_dataset_path, seq_index_col, target_col,
                                                                               test_size, random_state)
        real_f1 = self.tune_classifier(X_train_real, X_test_real, y_train_real, y_test_real, param_grid)
        print(f"[+] Processed {real_dataset_path}: F1-ratio = {real_f1}")

        f1_ratios = []
        for synthetic_file in filter(lambda f: f.endswith('.csv'), os.listdir(synthetic_folder_path)):
            synth_path = os.path.join(synthetic_folder_path, synthetic_file)
            X_train_synth, X_test_synth, y_train_synth, y_test_synth = self.split_data(synth_path, seq_index_col,
                                                                                       target_col, test_size, random_state)
            synth_f1 = self.tune_classifier(X_train_synth, X_test_real, y_train_synth, y_test_real, param_grid)
            epoch = int(os.path.splitext(synthetic_file)[0])
            f1_ratios.append({'Epochs': epoch, 'F1-ratio': synth_f1 / real_f1})
            print(f"[+] Processed {synthetic_file}: F1-ratio = {synth_f1 / real_f1}")

        results_df = pd.DataFrame(f1_ratios).sort_values(by="Epochs").reset_index(drop=True)
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, "f1_ratios.csv")
        results_df.to_csv(output_csv, index=False)
        print(f"[+] Classification results saved to '{output_csv}'.")
        return results_df

    def load_and_process_data(self, file_path: str, seq_index_col: str = 'SID', target_col: str = None):
        """
        Load and process a dataset, grouping by a sequence index and extracting features and labels.

        Args:
            file_path (str): Path to the dataset CSV file.
            seq_index_col (str): Column identifying sequences (default 'SID').
            target_col (str): Column name for the target label.

        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target labels array.
        """
        try:
            df = pd.read_csv(file_path)
            grouped = df.groupby(seq_index_col)
            X, y = [], []
            for _, group in grouped:
                features = group.iloc[:, 1:-1].values
                label = group[target_col].mode()[0]
                X.append(features)
                y.append(label)
            return np.array(X), np.array(y)
        except FileNotFoundError:
            print(f"[-] Error: File not found: {file_path}")
        except pd.errors.EmptyDataError:
            print(f"[-] Error: File is empty: {file_path}")
        except Exception as e:
            print(f"[-] Error loading data from {file_path}: {e}")
        return None, None

    def split_data(self, dataset_path: str, seq_col: str = 'SID', target_col: str = None,
                   test_size: float = 0.2, random_state: int = 0):
        """
        Split data into training and test sets.

        Args:
            dataset_path (str): Path to the dataset CSV file.
            seq_col (str): Column identifying sequences (default 'SID').
            target_col (str): Column name for the target label.
            test_size (float): Proportion of the dataset for the test set.
            random_state (int): Random state for reproducibility.

        Returns:
            tuple: (X_train, X_test, y_train, y_test).
        """
        try:
            X, y = self.load_and_process_data(dataset_path, seq_col, target_col)
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        except Exception as e:
            print(f"[-] Error splitting data: {e}")
        return None, None, None, None

    def tune_classifier(self, X_train, X_test, y_train, y_test, param_grid):
        """
        Tune the KNeighborsTimeSeriesClassifier using GridSearchCV.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training target labels.
            param_grid (dict): Hyperparameter grid for GridSearchCV.

        Returns:
            KNeighborsTimeSeriesClassifier: Best classifier found during tuning.
        """
        try:
            classifier = KNeighborsTimeSeriesClassifier()
            grid_search = GridSearchCV(
                estimator=classifier,
                param_grid=param_grid,
                scoring="f1_weighted",
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_classifier = grid_search.best_estimator_
            # Predict using the best classifier
            y_pred = best_classifier.predict(X_test)
            # Compute the evaluation metric for the test set
            score = f1_score(y_test, y_pred, average='weighted')
            return score
        except Exception as e:
            print(f"[-] Error tuning classifier: {e}")
        return None
