import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer

def load_and_split_data(real_data_path, test_size=0.2, random_state=0):
    """
    Load real data from a CSV file and split it into training and test sets.

    :param real_data_path: Path to the CSV file containing the real data.
    :param test_size: Proportion of the data to be used for testing.
    :param random_state: Random seed for reproducibility of splits.
    :return: Tuple containing the training and test datasets.
    """
    # Load real data from CSV file
    real_data = pd.read_csv(real_data_path)
    
    # Separate features and target variable
    x = real_data.iloc[:, :-1]
    y = real_data.iloc[:, -1]
    
    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    # Combine features and target into train and test DataFrames
    train_dataset = pd.concat([x_train, y_train], axis=1)
    test_dataset = pd.concat([x_test, y_test], axis=1)
    
    return train_dataset, test_dataset

def save_data(train_dataset, test_dataset, train_path, test_path):
    """
    Save the train and test datasets to CSV files.

    :param train_dataset: The training dataset to save.
    :param test_dataset: The test dataset to save.
    :param train_path: Path where the training dataset will be saved.
    :param test_path: Path where the test dataset will be saved.
    """
    # Save the datasets to CSV files
    train_dataset.to_csv(train_path, index=False)
    test_dataset.to_csv(test_path, index=False)

def initialize_synthesizer(metadata_path, epochs=60, cuda=True, verbose=True):
    """
    Initialize the PARSynthesizer with the provided metadata and configuration.

    :param metadata_path: Path to the metadata JSON file.
    :param epochs: Number of training epochs for the synthesizer.
    :param cuda: Whether to use GPU for training.
    :param verbose: Whether to print training progress.
    :return: An instance of the PARSynthesizer.
    """
    # Load metadata from JSON file
    metadata = Metadata.load_from_json(metadata_path)
    
    # Initialize the PARSynthesizer
    synthesizer = PARSynthesizer(
        metadata,
        enforce_min_max_values=True,   # Ensure synthetic data respects real data min/max boundaries
        enforce_rounding=False,        # Maintain the same decimal precision as the real data
        locales=['en_US'],             # Define locales for PII columns
        context_columns=['harsh_event'], # Define columns that should remain constant in sequences
        epochs=epochs,                 # Set the number of training epochs
        cuda=cuda,                     # Enable GPU usage for faster training
        verbose=verbose                # Print training progress
    )
    
    return synthesizer

def train_synthesizer(synthesizer, train_dataset, save_path='Synthesizer.pkl'):
    """
    Train the synthesizer on the real training dataset and save the trained model.

    :param synthesizer: The synthesizer object to train.
    :param train_dataset: The dataset used to train the synthesizer.
    :param save_path: Path to save the trained synthesizer model.
    """
    # Train the synthesizer on the real training dataset
    synthesizer.fit(train_dataset)
    
    # Save the trained synthesizer model
    synthesizer.save(filepath=save_path)
    print(f"Synthesizer model saved to {save_path}")

def generate_synthetic_data(synthesizer, num_sequences=55, sequence_length=5600, output_path='synthetic_data.csv'):
    """
    Generate synthetic data using the trained synthesizer and save it to a CSV file.

    :param synthesizer: The trained synthesizer to generate synthetic data.
    :param num_sequences: Number of sequences to generate.
    :param sequence_length: Length of each generated sequence.
    :param output_path: Path to save the generated synthetic data.
    :return: The generated synthetic data as a DataFrame.
    """
    # Generate synthetic data
    synthetic_data = synthesizer.sample(num_sequences=num_sequences, sequence_length=sequence_length)
    
    # Save the synthetic data to CSV
    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")
    
    return synthetic_data
