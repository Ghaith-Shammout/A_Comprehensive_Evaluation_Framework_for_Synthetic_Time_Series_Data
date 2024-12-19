import pandas as pd
import os
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer

class SyntheticDataGenerator:
    def __init__(self):
        self.metadata = None
        self.synthesizer = None

    def define_metadata(self, dataset, seq_key, seq_index, date_format, metadata_path):
        """
        Defines and saves metadata for a given dataset.

        Args:
            dataset (str): File path to the dataset (CSV format).
            seq_key (str): Column identifying a sequence in the dataset.
            seq_index (str): Column determining the spacing between rows in a sequence.
            date_format (str): Date format stored in the file (e.g., %d%m%Y).
            metadata_path (str): Path to save the metadata JSON file.

        Returns:
            None
        """
        try:
            # Load dataset
            df = pd.read_csv(dataset)
            
            # Detect metadata from DataFrame
            self.metadata = Metadata.detect_from_dataframe(df)
            
            # Update metadata for sequence key and index
            self.metadata.update_column(column_name=seq_key, sdtype='id')
            self.metadata.update_column(column_name=seq_index, sdtype='datetime', datetime_format=date_format)
            self.metadata.set_sequence_key(seq_key)
            self.metadata.set_sequence_index(seq_index)
            
            # Check if the output directory exists
            output_dir = os.path.dirname(metadata_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)  # Create the directory if it doesn't exist
                
            # Save metadata to a JSON file
            self.metadata.save_to_json(metadata_path)
            print(f"[+] Metadata successfully defined and saved to {metadata_path}")
        
        except Exception as e:
            print(f"[-] Error defining metadata: {str(e)}")
            raise

    def initialize_synthesizer(self, metadata_path, context_columns, epochs, verbose, cuda):
        """
        Initializes the PARSynthesizer.

        Args:
            metadata_path (str): Path to the metadata JSON file.
            context_columns (list): Columns to be used as context for the synthesizer.
            epochs (int): Number of training epochs.
            verbose (bool): Whether to display training progress.
            cuda (bool): Whether to use GPU for training.

        Returns:
            None
        """
        try:
            self.metadata = Metadata.load_from_json(metadata_path)
            self.synthesizer = PARSynthesizer(
                metadata=self.metadata,
                context_columns=context_columns,
                epochs=epochs,
                verbose=verbose,
                cuda=cuda
            )
            print("[+] Synthesizer initialized successfully")
            return self.synthesizer

        except Exception as e:
            print(f"[-] Error initializing synthesizer: {str(e)}")
            raise

    def train_synthesizer(self, train_dataset, save_path):
        """
        Trains the synthesizer on the given dataset and saves the model.

        Args:
            train_dataset (str): Path to the training dataset (CSV format).
            save_path (str): Path to save the trained synthesizer model.

        Returns:
            None
        """
        try:
            dataset = pd.read_csv(train_dataset)
            print("[+] Preparing the model for training")
            self.synthesizer.fit(dataset)
            self.synthesizer.save(save_path)
            print(f"[+] Synthesizer model saved to {save_path}")
        except Exception as e:
            print(f"[-] Error training synthesizer: {str(e)}")
            raise

    def generate_synthetic_data(self, num_sequences, sequence_length, output_path, seq_key, seq_index):
        """
        Generates synthetic data and saves it to a file.
    
        Args:
            num_sequences (int): Number of synthetic sequences to generate.
            sequence_length (int): Length of each synthetic sequence.
            output_path (str): Path to save the generated synthetic data.
            seq_key (str): The primary column to sort by.
            seq_index (str): The secondary column to sort by.
    
        Returns:
            None
        """
        try:
            # Generate synthetic data
            synthetic_data = self.synthesizer.sample(num_sequences=num_sequences, sequence_length=sequence_length)
    
            # Sort the data by the provided columns
            synthetic_data = synthetic_data.sort_values(by=[seq_key, seq_index], ascending=True)
    
            # Save the sorted data to a CSV file
            synthetic_data.to_csv(output_path, index=False)
            
            # Print a message with the column names used for sorting
            print(f"[+] Synthetic data sorted by {seq_key} and {seq_index} saved to {output_path}")
            
        except Exception as e:
            print(f"[-] Error generating synthetic data: {str(e)}")
            raise


        