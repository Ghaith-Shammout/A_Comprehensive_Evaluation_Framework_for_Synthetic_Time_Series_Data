import logging
import yaml
from preprocessing import Preprocessing
from generating import SyntheticDataGenerator
from evaluation import PopulationFidelity
from classification import Classifier
from correlation_analysis import CorrelationAnalysis


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
        logging.info("Phase 1: Data Preprocessing Started.")
        # Create an instance of the Preprocessing class
        preprocessor = Preprocessing(
            input_file=config['preprocessing']['input_file'],
            output_file=config['preprocessing']['output_file'],
        )
        
        # running preprocessing steps with their respective arguments
        preprocessor.remove_unwanted_columns(unwanted_columns=config['preprocessing']['unwanted_columns'])
        preprocessor.enforce_date_format(date_column=config['preprocessing']['date_column'],
                                         date_format=config['preprocessing']['date_format'])
        preprocessor.normalize_data(method=config['preprocessing']['normalization_method'], 
                                   normalize_columns=config['preprocessing']['normalize_columns'])
        preprocessor.label_encode_and_save(categorical_columns=config['preprocessing']['categorical_columns'])
        preprocessor.sliding_window(window_size=config['preprocessing']['window_size'],
                                    step_size=config['preprocessing']['step_size'])
        logging.info("Phase 1 Completed Successfully.")

      
        # Phase 2: Synthetic Data Generation 
        logging.info("Phase 2: Synthetic Data Generation Process Started.")
        
        # Define a generator
        generator = SyntheticDataGenerator()
        
        logging.info("Phase 2.1: Metadata Definition started.")
        metadata = generator.define_metadata(
            dataset=config['preprocessing']['output_file'],
            seq_key=config['metadata']['seq_key'],  # Sequence key
            seq_index=config['metadata']['seq_index'],      # Sequence index
            date_format=config['metadata']['date_format'],
            metadata_path=config['metadata']['metadata_path']
        )
        logging.info("Phase 2.1: Metadata Definition Completed Successfully.")

        for epoch in config['synthesizer']['epochs']:
            logging.info(f"Starting pipeline run with {epoch} epochs.")

            # Update the number of epochs in the configuration dynamically
            #config['synthesizer']['epochs'] = epochs
            logging.info(f"Phase 2.2: Synthesizer Initializing with {epoch} epochs Started.")
            # Train the synthesizer
            synthesizer = generator.initialize_synthesizer(
                metadata_path=config['metadata']['metadata_path'],
                context_columns=config['synthesizer']['context_columns'],
                epochs=epoch,
                verbose=config['synthesizer']['verbose'],
                cuda=config['synthesizer']['cuda']
            )
            logging.info(f"Phase 2.2: Synthesizer Initializing with {epoch} epochs Successfully Completed.")

            # Phase 2.3: Synthesizer Training
            logging.info(f"Phase 2.3: Synthesizer Training with {epoch} epochs Started.")
            generator.train_synthesizer(
                #synthesizer=synthesizer,
                train_dataset=config['preprocessing']['output_file'],  # Ensure DataFrame is passed
                save_path=f"outputs/Models/synthesizer_{epoch}_epochs.pkl"
            )
            logging.info(f"Phase 2.3: Synthesizer Training with {epoch} epochs Completed.")

            # Phase 2.4: Synthetic Data Generation
            logging.info(f"Phase 2.4: Synthetic Data Generation with {epoch} epochs Started.")
            logging.info("Phase 4: Synthetic Data Generation started.")
            generator.generate_synthetic_data(
                #synthesizer=synthesizer,
                num_sequences=config['generation']['num_sequences'],
                sequence_length=config['generation']['sequence_length'],
                output_path=f"outputs/Synth_Data/{epoch}.csv",
                seq_key=config['metadata']['seq_key'],          # Sequence key used to sort by
                seq_index=config['metadata']['seq_index'],      # Sequence index used to sort by
            )
            logging.info(f"Phase 2.4: Synthetic Data Generation with {epoch} epochs Completed.")

            
        # Phase 3: Synthetic Data Evaluation
        logging.info(f"Phase 3: Synthetic Data Evaluation Started.")
        
        # Define an evaluator
        real_data_path = config['preprocessing']['output_file']
        synth_folder = config['evaluation']['synth_folder']
        exclude_cols = config['evaluation']['exclude_cols']
        sequence_length = config['generation']['sequence_length']
        evaluator = PopulationFidelity(real_data_path, synth_folder, exclude_cols, sequence_length)

        msas_output = config['evaluation']['msas_output']
        evaluator.msas(msas_output)

        awd_output = config['evaluation']['awd_output']
        evaluator.awd(awd_output)
        
        logging.info(f"Phase 3: Synthetic Data Evaluation Completed.")
    
        # Phase 4: Classification Process
        logging.info(f"Phase 4: Classification Process Started.")
        classifier = Classifier()

        real_dataset_path = config['preprocessing']['output_file']
        synthetic_folder_path = config['evaluation']['synth_folder']
        seq_index_col = config['classification']['seq_index_col']
        target_col = config['classification']['target_col']
        metric = config['classification']['metric']
        param_grid = config['classification']['param_grid']
        test_size = config['classification']['test_size']
        random_state = config['classification']['random_state']
        
        classifier.classify(real_dataset_path, synthetic_folder_path,
                           seq_index_col, target_col, metric, param_grid,
                           test_size, random_state)
        
        logging.info(f"Phase 4: Classification Process Completed.")

        
        # Phase 5: Correlation Analysis
        logging.info(f"Phase 5: Correlation Analysis Started.")
        
        # Load paths from config
        f1_ratios_path = config['correlation_analysis']['f1_ratios_path']
        metric_file_paths = config['correlation_analysis']['metric_file_paths']
        metric_columns = config['correlation_analysis']['metric_columns']  # List of metrics like ["MSAS", "AWD"]
        plot_output_path = config['correlation_analysis']['plot_output_path']
        correlation_output_path = './outputs/correlation_results.csv'  # Path to save the CSV with correlation results
        
        # Instantiate the CorrelationAnalysis class
        analysis = CorrelationAnalysis()
        
        # Perform correlation analysis for all metrics and save results
        analysis.correlation_analysis(
            f1_ratios_path = f1_ratios_path,
            metric_columns=metric_columns,
            metric_file_paths=metric_file_paths,  # Pass the metric file paths dictionary
            output_dir=plot_output_path,
            correlation_output_path=correlation_output_path
        )

        logging.info(f"Phase 5: Correlation Analysis Completed.")

        
        logging.info("Pipeline execution completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
