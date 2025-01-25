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

  def compute(self, real_dataset_path, synthetic_folder_path, seq_index_col, target_col,
              param_grid, test_size, random_state, output_dir):
    """
    Compute F1-ratios for real and synthetic datasets.
  
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
  
    # Process the real dataset
    try:
        X_train_real, X_test_real, y_train_real, y_test_real = self.split_data(
            real_dataset_path, seq_index_col, target_col, test_size, random_state
        )
        real_f1 = self.tune_classifier(X_train_real, X_test_real, y_train_real, y_test_real, param_grid)
        print(f"[+] Processed {real_dataset_path}: F1-score = {real_f1}")
    except Exception as e:
        raise RuntimeError(f"Error processing real dataset: {e}")
  
    # Handle division by zero if real_f1 is 0
    if real_f1 == 0:
        raise ValueError("F1-score for the real dataset is 0. Cannot compute F1-ratios.")
  
    # Process synthetic datasets
    f1_ratios = []
    for synthetic_file in filter(lambda f: f.endswith('.csv'), os.listdir(synthetic_folder_path)):
        synth_path = os.path.join(synthetic_folder_path, synthetic_file)
        try:
            # Split synthetic data
            X_train_synth, _, y_train_synth, _ = self.split_data(
                synth_path, seq_index_col, target_col, test_size, random_state
            )
            # Compute F1-score for synthetic data
            synth_f1 = self.tune_classifier(X_train_synth, X_test_real, y_train_synth, y_test_real, param_grid)
            # Calculate F1-ratio
            f1_ratio = synth_f1 / real_f1
            copy = int(os.path.splitext(synthetic_file)[0].split('_')[-1])  # Extract copy number from filename
            f1_ratios.append({'Copy': copy, 'F1-ratio': f1_ratio})
            print(f"[+] Processed {synthetic_file}: F1-ratio = {f1_ratio}")
        except Exception as e:
            print(f"[-] Error processing {synthetic_file}: {e}")
  
    # Save results to a CSV file
    results_df = pd.DataFrame(f1_ratios).sort_values(by="Copy").reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "f1", f"{os.path.basename(synthetic_folder_path)}.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # Ensure the directory exists
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
        # Load the dataset
        df = pd.read_csv(file_path)

        # Check if the target column is provided
        if target_col is None:
            raise ValueError("Target column must be specified.")

        # Validate that the sequence index and target columns exist in the dataset
        if seq_index_col not in df.columns:
            raise ValueError(f"Sequence index column '{seq_index_col}' not found in the dataset.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the dataset.")

        # Drop the date column (if it exists)
        if 'date' in df.columns:
            df = df.drop(columns=['date'])

        # Convert feature columns to numeric, coercing errors to NaN
        feature_columns = [col for col in df.columns if col not in [seq_index_col, target_col]]
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
          
        # Group the data by the sequence index column
        grouped = df.groupby(seq_index_col)

        # Initialize lists to store features and labels
        X, y = [], []

        # Process each group
        for group_name, group in grouped:
            # Extract features (all columns except the sequence index and target column)
            #features = group[feature_columns].values
            features = group.iloc[:, 1:-1].values
            # Extract the mode of the target column as the label for the group
            label = group[target_col].mode()[0]
            X.append(features)
            y.append(label)

        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Validate the shapes of X and y
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatch in the number of samples: X has {X.shape[0]} samples, y has {y.shape[0]} samples.")
        if X.shape[2] != len(feature_columns):
            raise ValueError(f"Mismatch in the number of features: Expected {len(feature_columns)}, got {X.shape[1]}.")

        # Log the shapes for debugging
        #print(f"[+] Processed dataset: X shape = {X.shape}, y shape = {y.shape}")
        #print(f"[+] Feature columns used: {feature_columns}")
        #print(f"[+] Target column used: {target_col}")

        return X, y
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Error: File is empty: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading data from {file_path}: {e}")
    
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
