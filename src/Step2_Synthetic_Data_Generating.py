import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer

def define_metadata(dataset, seq_key, seq_index):
    """
    Defines and returns metadata for a given dataset.
    
    Args:
        dataset (DataFrame): The dataset (assumed to be a pandas DataFrame) to extract metadata from.
        seq_key (str): The name of the column that identifies a sequence within the dataset.
        seq_index (str): The name of the column that determines the spacing between rows in a sequence.
        
    Returns:
        Metadata: An object that contains metadata, including sequence key and index information.
        
    Raises:
        ValueError: If the dataset does not contain the provided `seq_key` or `seq_index` columns.
        TypeError: If the dataset is not a valid DataFrame.
        AttributeError: If the `Metadata.detect_from_dataframes` method is not available or fails.
    """
    try:
        # Ensure the dataset is a DataFrame
        if not hasattr(dataset, 'columns'):
            raise TypeError("The provided dataset is not a valid DataFrame.")
        
        # Attempt to detect metadata from the dataset
        metadata = Metadata.detect_from_dataframes(dataset)
        
        # Ensure the detected metadata object has the necessary methods
        if not hasattr(metadata, 'set_sequence_key') or not hasattr(metadata, 'set_sequence_index'):
            raise AttributeError("Detected metadata object does not have expected methods.")
        
        # Set the sequence key in the metadata
        metadata.set_sequence_key(column_name=seq_key)  # column that identifies a sequence in the dataset
        
        # Set the sequence index in the metadata
        metadata.set_sequence_index(column_name=seq_index)  # column that determines the spacing between rows in a sequence.
        
        return metadata

    except ValueError as e:
        # Handle case where the sequence columns are not found
        raise ValueError(f"Error with the provided column names: {e}")
    except TypeError as e:
        # Handle invalid dataset input
        raise TypeError(f"Dataset error: {e}")
    except AttributeError as e:
        # Handle issues with the metadata detection or methods
        raise AttributeError(f"Metadata detection error: {e}")
    except Exception as e:
        # Catch all other exceptions
        raise Exception(f"An unexpected error occurred: {e}")

    

def initialize_synthesizer(metadata_path, context_columns, epochs=60, sample_size=1, cuda=True, verbose=True):
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
        context_columns=context_columns, # Define columns that should remain constant in sequences
        epochs=epochs,                 # Set the number of training epochs
        sample_size=sample_size,       # The number of times to sample before choosing and returning a sample. 
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
