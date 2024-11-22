from sdv.sequential import PARSynthesizer
from sdv.metadata import Metadata

def initialize_synthesizer(metadata_path, context_columns, epochs, cuda):
    metadata = Metadata.load_from_json(metadata_path)
    synthesizer = PARSynthesizer(
        metadata=metadata,
        context_columns=context_columns,
        epochs=epochs,
        cuda=cuda
    )
    return synthesizer

def train_synthesizer(synthesizer, train_dataset, save_path):
    synthesizer.fit(train_dataset)
    synthesizer.save(save_path)
    print(f"[+] Synthesizer model saved to {save_path}")
