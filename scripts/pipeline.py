import logging
import yaml
from scripts.preprocessing import preprocess_data
from scripts.metadata import define_metadata
from scripts.training import initialize_synthesizer, train_synthesizer
from scripts.generation import generate_synthetic_data
from scripts.evaluation import evaluate_synthetic_data

def setup_logging(log_file, level):
    """
    Configures logging for the pipeline.
    """
    logging.basicConfig(
        filename=log_file,
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def load_config(config_path):
    """
    Loads the pipeline configuration from a YAML file.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    """
    Main pipeline integrating all phases.
    """
    # Load configuration
    config = load_config("config/pipeline_config.yaml")
    setup_logging(config['logging']['log_file'], config['logging']['level'])
    logging.info("Pipeline execution started.")

    try:
        # Phase 1: Data Preprocessing
        logging.info("Phase 1: Data Preprocessing started.")
        preprocess_data(
            input_path=config['data']['input_path'],
            output_path=config['data']['input_path'],  # Overwrite with preprocessed data
            remove_columns=config['data']['preprocessing']['remove_columns'],
            normalization_method=config['data']['preprocessing']['normalization']
        )
        logging.info("Phase 1 completed successfully.")

        # Phase 2: Metadata Definition
        logging.info("Phase 2: Metadata Definition started.")
        metadata = define_metadata(
            dataset_path=config['data']['input_path'],
            seq_key="Load_Type",  # Sequence key
            seq_index="date"      # Sequence index
        )
        metadata.save(config['data']['metadata_path'])
        logging.info("Phase 2 completed successfully.")

        # Phase 3: Model Training
        logging.info("Phase 3: Model Training started.")
        synthesizer = initialize_synthesizer(
            metadata_path=config['data']['metadata_path'],
            context_columns=config['synthesizer']['context_columns'],
            epochs=config['synthesizer']['epochs'],
            cuda=config['synthesizer']['cuda']
        )
        train_synthesizer(
            synthesizer=synthesizer,
            train_dataset=config['data']['input_path'],
            save_path="data/Synthesizer.pkl"
        )
        logging.info("Phase 3 completed successfully.")

        # Phase 4: Synthetic Data Generation
        logging.info("Phase 4: Synthetic Data Generation started.")
        generate_synthetic_data(
            synthesizer=synthesizer,
            num_sequences=config['generation']['num_sequences'],
            sequence_length=config['generation']['sequence_length'],
            output_path=config['data']['output_path']
        )
        logging.info("Phase 4 completed successfully.")

        # Phase 5: Analysis and Evaluation
        logging.info("Phase 5: Analysis and Evaluation started.")
        evaluate_synthetic_data(
            real_data_path=config['data']['input_path'],
            synthetic_data_path=config['data']['output_path'],
            metrics=config['evaluation']['metrics']
        )
        logging.info("Phase 5 completed successfully.")

        logging.info("Pipeline execution completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
