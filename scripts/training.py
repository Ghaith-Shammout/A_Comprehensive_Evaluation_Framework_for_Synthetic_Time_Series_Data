import pandas as pd
from sdv.sequential import PARSynthesizer
from sdv.metadata import Metadata

def initialize_synthesizer(metadata_path, context_columns, epochs, verbose, cuda):
    metadata = Metadata.load_from_json(metadata_path)
    synthesizer = PARSynthesizer(
        metadata=metadata,
        context_columns=context_columns,
        epochs=epochs,
        verbose=verbose,
        cuda=cuda
    )
    return synthesizer

def train_synthesizer(synthesizer, train_dataset, save_path):
    dataset = pd.read_csv(train_dataset)
    print(f"[+] Prepare the model to fit")
    synthesizer.fit(dataset)
    synthesizer.save(save_path)
    print(f"[+] Synthesizer model saved to {save_path}")
