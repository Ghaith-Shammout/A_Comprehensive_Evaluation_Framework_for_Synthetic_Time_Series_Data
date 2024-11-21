import numpy as np
import pandas as pd
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
    
    Args:
        metadata_path (str): Path to the metadata JSON file.
        context_columns (list): List of columns to remain constant in sequences.
        epochs (int, optional): Number of training epochs for the synthesizer (default is 60).
        sample_size (int, optional): The number of samples to choose from before returning a sample (default is 1).
        cuda (bool, optional): Whether to use GPU for training (default is True).
        verbose (bool, optional): Whether to print training progress (default is True).
        
    Returns:
        PARSynthesizer: An instance of the PARSynthesizer initialized with the given parameters.
        
    Raises:
        FileNotFoundError: If the metadata file at `metadata_path` is not found.
        ValueError: If there is an issue with the provided metadata or configuration.
        TypeError: If the input parameters are of incorrect types.
        Exception: For any other unexpected errors.
    """
    try:
        # Ensure metadata_path is a string and context_columns is a list
        if not isinstance(metadata_path, str):
            raise TypeError("The metadata path should be a string.")
        if not isinstance(context_columns, list):
            raise TypeError("The context_columns should be a list.")
        
        # Load metadata from JSON file
        metadata = Metadata.load_from_json(metadata_path)  # Attempt to load metadata
        
        # Validate that metadata is loaded correctly
        if not metadata:
            raise ValueError(f"Failed to load metadata from the file: {metadata_path}")
        
        # Initialize the PARSynthesizer with the provided parameters
        synthesizer = PARSynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,  # Ensure synthetic data respects real data min/max boundaries
            enforce_rounding=False,       # Maintain the same decimal precision as the real data
            locales=['en_US'],            # Define locales for PII columns (Personally Identifiable Information)
            context_columns=context_columns,  # Columns that should remain constant in sequences
            epochs=epochs,                # Set the number of training epochs
            sample_size=sample_size,      # The number of samples to select before choosing a result
            cuda=cuda,                    # Whether to use GPU for faster training
            verbose=verbose               # Whether to print training progress
        )
        
        return synthesizer  # Return the initialized synthesizer

    except FileNotFoundError:
        # Handle file not found error if metadata_path is incorrect
        raise FileNotFoundError(f"The metadata file at {metadata_path} was not found.")
    except ValueError as e:
        # Handle value errors that may arise from invalid metadata or parameter issues
        raise ValueError(f"Error with metadata or configuration: {e}")
    except TypeError as e:
        # Handle type errors (invalid parameter types)
        raise TypeError(f"Invalid parameter type: {e}")
    except Exception as e:
        # Catch all other exceptions and provide context
        raise Exception(f"An unexpected error occurred during synthesizer initialization: {e}")


def train_synthesizer(synthesizer, train_dataset, save_path='Synthesizer.pkl'):
    """
    Train the synthesizer on the real training dataset and save the trained model.

    :param synthesizer: The synthesizer object to train.
    :param train_dataset: The dataset used to train the synthesizer.
    :param save_path: Path to save the trained synthesizer model.
    """
     # Train the synthesizer on the real training dataset
    synthesizer.fit(train_dataset)
    print("[+] Training complete.")
    
    # Save the trained synthesizer model
    synthesizer.save(filepath=save_path)
    print(f"[+] Synthesizer model saved to {save_path}")

def generate_synthetic_data(synthesizer, num_sequences=1457, sequence_length=96, output_path='synthetic_data.csv'):
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
    print("[+] Synthetic data successfully generated.")
    
    # Save the synthetic data to CSV
    synthetic_data.to_csv(output_path, index=False)
    print(f"[+] Synthetic data saved to {output_path}")
    
    return synthetic_data
