from sdv.metadata import Metadata

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

    dataset = pd.read_csv(dataset_path)
    metadata = Metadata.detect_from_dataframe(dataset)
    metadata.update_column(
        column_name=seq_key,
        sdtype='id')
    metadata.set_sequence_key(seq_key)
    metadata.set_sequence_index(seq_index)
    return metadata
    
