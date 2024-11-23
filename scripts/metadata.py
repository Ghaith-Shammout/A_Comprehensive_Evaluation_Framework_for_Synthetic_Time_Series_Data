import pandas as pd
from sdv.metadata import Metadata

def define_metadata(dataset, seq_key, seq_index, date_format):
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
    df = pd.read_csv(dataset)
    metadata = Metadata.detect_from_dataframe(df)
    metadata.update_column(
        column_name=seq_key,
        sdtype='id')
    
    metadata.update_column(
        column_name=seq_index,
        sdtype='datetime',
        datetime_format=date_format)

    metadata.set_sequence_key(seq_key)
    metadata.set_sequence_index(seq_index)
    print(f"[+] Metadata successfully defined")
    return metadata
    
