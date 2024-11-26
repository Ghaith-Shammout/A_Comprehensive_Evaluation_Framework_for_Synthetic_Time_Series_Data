import logging
import yaml
from preprocessing import preprocess_data
from metadata import define_metadata
from training import initialize_synthesizer, train_synthesizer
from generation import generate_synthetic_data
from evaluation import population_fidelity_measure

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
            input_file=config['preprocessing']['input_file'],
            output_file=config['preprocessing']['output_file'],
            unwanted_columns=config['preprocessing']['unwanted_columns'],
            date_column=config['preprocessing']['date_column'],
            categorical_columns=config['preprocessing']['categorical_columns'],
            window_size=config['preprocessing']['window_size'],
            step_size=config['preprocessing']['step_size'],
            normalization_method=config['preprocessing']['normalization_method']
        )
        logging.info("Phase 1 completed successfully.")

        # Phase 2: Metadata Definition
        logging.info("Phase 2: Metadata Definition started.")
        metadata = define_metadata(
            dataset=config['preprocessing']['output_file'],
            seq_key=config['metadata']['seq_key'],  # Sequence key
            seq_index=config['metadata']['seq_index'],      # Sequence index
            date_format=config['metadata']['date_format']
        )
        metadata.save_to_json(config['metadata']['metadata_path'])
        logging.info("Phase 2 completed successfully.")

        """
        # Phase 3: Model Training
        logging.info("Phase 3: Model Training started.")
        synthesizer = initialize_synthesizer(
            metadata_path=config['metadata']['metadata_path'],
            context_columns=config['synthesizer']['context_columns'],
            epochs=config['synthesizer']['epochs'],
            verbose=config['synthesizer']['verbose'],
            cuda=config['synthesizer']['cuda']
        )
        train_synthesizer(
            synthesizer=synthesizer,
            train_dataset=config['preprocessing']['output_file'],
            save_path=config['synthesizer']['model_output']
        )
        logging.info("Phase 3 completed successfully.")
        """
        
        for epochs in range(100, 600, 100):
            logging.info(f"Starting pipeline run with {epochs} epochs.")

            # Update the number of epochs in the configuration dynamically
            config['synthesizer']['epochs'] = epochs

            # Train the synthesizer
            synthesizer = initialize_synthesizer(
                metadata_path=config['metadata']['metadata_path'],
                context_columns=config['synthesizer']['context_columns'],
                epochs=config['synthesizer']['epochs'],
                verbose=config['synthesizer']['verbose'],
                cuda=config['synthesizer']['cuda']
            )
            train_synthesizer(
                synthesizer=synthesizer,
                train_dataset=config['preprocessing']['output_file'],  # Ensure DataFrame is passed
                save_path=f"outputs/Models/synthesizer_{epochs}_epochs.pkl"
            )
            logging.info(f"Training a model with {epochs} epochs completed.")

            # Phase 4: Synthetic Data Generation
            logging.info("Phase 4: Synthetic Data Generation started.")
            generate_synthetic_data(
                synthesizer=synthesizer,
                num_sequences=config['generation']['num_sequences'],
                sequence_length=config['generation']['sequence_length'],
                output_path=f"outputs/Synth_Data/synth_{epochs}_epochs.csv"
            )
            logging.info(f"Generating synthetic data with {epochs} completed.")

        
        # Phase 5: Analysis and Evaluation
        logging.info("Phase 5: Analysis and Evaluation started.")
        population_fidelity_measure(
            real_file=config['preprocessing']['output_file'],
            synth_folder=config['evaluation']['synth_folder'],
        )
        logging.info("Phase 5 completed successfully.")
        

        logging.info("Pipeline execution completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
