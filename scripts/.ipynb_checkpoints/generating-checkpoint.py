import os
import pandas as pd
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer


class SyntheticDataGenerator:
    """
    A class for generating synthetic data using the SDV framework, including 
    metadata definition, model initialization, training, and synthetic data generation.
    """

    def __init__(self):
        self.metadata = None
        self.synthesizer = None

    def define_metadata(self, dataset_path: str, sequence_key: str, sequence_index: str, 
                        date_format: str, metadata_path: str):
        """
        Define and save metadata for a given dataset.

        Args:
            dataset_path (str): Path to the input dataset (CSV format).
            sequence_key (str): Column identifying sequences in the dataset.
            sequence_index (str): Column representing sequence order or datetime.
            date_format (str): Format of datetime values in the dataset.
            metadata_path (str): Path to save the metadata JSON file.
        """
        try:
            if os.path.exists(metadata_path):
              self.metadata = Metadata.load_from_json(metadata_path)
              return
            else:
              df = pd.read_csv(dataset_path)
              self.metadata = Metadata.detect_from_dataframe(df)
              self.metadata.update_column(column_name=sequence_key, sdtype='id')
              self.metadata.update_column(column_name=sequence_index, sdtype='datetime', datetime_format=date_format)
              self.metadata.set_sequence_key(sequence_key)
              self.metadata.set_sequence_index(sequence_index)
              self.metadata.save_to_json(metadata_path)
              print(f"[+] Metadata defined and saved to '{metadata_path}'.")
        except Exception as e:
            print(f"[-] Error defining metadata: {e}")
            raise


    def load_synthesizer(self, models_dir: str, epochs: int = 300):
        """
        Load an existing synthesizer model for a specific epoch if it exists.
    
        Args:
            models_dir (str): Path to the directory containing model files.
            epochs (int, optional): Epoch number of the model to load. Default is 300.
    
        Returns:
            PARSynthesizer or None: Loaded synthesizer if the model exists; otherwise, None.
        """
        try:
            # Construct the expected model file name based on the epoch
            model_filename = f"{epochs}.pkl"
            model_path = os.path.join(models_dir, model_filename)
    
            if os.path.exists(model_path):
                print(f"[+] Found model for epoch {epochs} at '{model_path}'. Loading...")
                self.synthesizer = PARSynthesizer.load(model_path)
                print("[+] Synthesizer loaded successfully.")
                return self.synthesizer
            else:
                print(f"[-] No model found for epoch {epochs} at '{model_path}'.")
                return None
        except Exception as e:
            print(f"[-] Error in loading synthesizer: {e}")
            raise




    def initialize_synthesizer(self, metadata_path: str, context_columns: list[str], 
                               epochs: int = 300, verbose: bool = True, cuda: bool = False):
        """
        Initialize the PARSynthesizer with specified settings.

        Args:
            metadata_path (str): Path to the metadata JSON file.
            context_columns (list[str]): Columns used as context for the synthesizer.
            epochs (int, optional): Number of training epochs. Default is 300.
            verbose (bool, optional): Display training progress. Default is True.
            cuda (bool, optional): Use GPU for training. Default is False.
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
            print("[+] Synthesizer initialized successfully.")
        except Exception as e:
            print(f"[-] Error initializing synthesizer: {e}")
            raise

    def train_synthesizer(self, train_dataset_path: str, model_save_path: str):
        """
        Train the synthesizer on the provided dataset and save the trained model.

        Args:
            train_dataset_path (str): Path to the training dataset (CSV format).
            model_save_path (str): Path to save the trained synthesizer model.
        """
        try:
            dataset = pd.read_csv(train_dataset_path)
            print("[+] Training the synthesizer...")
            self.synthesizer.fit(dataset)
            self.synthesizer.save(model_save_path)
            print(f"[+] Trained synthesizer saved to '{model_save_path}'.")
        except Exception as e:
            print(f"[-] Error training synthesizer: {e}")
            raise

    def generate_synthetic_data(self, num_sequences: int, sequence_length: int, output_dir: str, 
                                sequence_key: str, sequence_index: str, num_files: int = 1):
        """
        Generate synthetic data and save it as CSV files.

        Args:
            num_sequences (int): Number of sequences to generate per file.
            sequence_length (int): Length of each sequence.
            output_dir (str): Directory to save the generated data files.
            sequence_key (str): Primary column for sorting the data.
            sequence_index (str): Secondary column for sorting the data.
            num_files (int, optional): Number of files to generate. Default is 1.
        """
        try:
            unique_id = 1  # Initialize a unique ID counter

            # used to get rid of error messgae
            # UserWarning: RNN module weights are not part of single contiguous chunk of memory
            self.synthesizer._model._model.rnn.flatten_parameters() 
            for i in range(1, num_files + 1):
                synthetic_data = self.synthesizer.sample(num_sequences=num_sequences, sequence_length=sequence_length)
                synthetic_data.sort_values(by=[sequence_key, sequence_index], ascending=True, inplace=True)

                # Assign auto-incrementing unique IDs based on sequence length
                synthetic_data['SID'] = [(unique_id + j // sequence_length) for j in range(len(synthetic_data))]
                unique_id += len(synthetic_data) // sequence_length

                output_path = f"{output_dir}/{i}.csv"
                synthetic_data.to_csv(output_path, index=False)
                print(f"[+] Synthetic data saved to '{output_path}'. Sorted by '{sequence_key}' and '{sequence_index}'.")
        except Exception as e:
            print(f"[-] Error generating synthetic data: {e}")
            raise