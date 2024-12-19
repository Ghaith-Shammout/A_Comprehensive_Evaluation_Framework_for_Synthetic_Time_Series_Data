import logging
import yaml
import os
from preprocessing import Preprocessing
from generating import SyntheticDataGenerator
from population_fidelity import PopulationFidelity
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
    # Path to the config directory
    config_dir = "config/"
    
    # Loop through all files in the directory
    for filename in os.listdir(config_dir):
        # Check if the file has a .yaml extension
        if filename.endswith(".yaml"):
            print(f"[+] Processing the configuration file: {filename}")
            
            # Construct the full file path
            config_file_path = os.path.join(config_dir, filename)
            
            # Load the configuration
            config = load_config(config_file_path)
            
            # Setup logging based on the config file
            setup_logging(config['logging']['log_file'], config['logging']['level'])
            
            # Log that the pipeline has started
            logging.info("Pipeline execution started.")   

            try:
                # Assuming source_file is like 'Steal.csv' from the config
                source_file = config['data']['source_file']
                
                # Extract the base name without the extension
                base_name = os.path.splitext(os.path.basename(source_file))[0]
                
                # Add the prefix "Real_" and reattach the extension
                real_file = f"./outputs/{base_name}/data/Real_{base_name}{os.path.splitext(source_file)[1]}"
                
                # Phase 1: Data Preprocessing
                logging.info("Phase 1: Data Preprocessing Started.")
                # Create an instance of the Preprocessing class
                preprocessor = Preprocessing(
                    input_file=source_file,
                    output_file=real_file,
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
                    dataset=real_file,
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
                    folder_path = config['synthesizer']['model_output']
                    
                    # Check if the directory exists, and if not, create it
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    generator.train_synthesizer(
                        #synthesizer=synthesizer,
                        train_dataset=real_file,  # Ensure DataFrame is passed                    
                        save_path = f"{folder_path}/{epoch}.pkl"
                        #save_path=f"{folder_path} + {epoch}.pkl"
                    )
                    logging.info(f"Phase 2.3: Synthesizer Training with {epoch} epochs Completed.")
        
                    # Phase 2.4: Synthetic Data Generation
                    logging.info(f"Phase 2.4: Synthetic Data Generation with {epoch} epochs Started.")
                    logging.info("Phase 4: Synthetic Data Generation started.")
                    
                    folder_path = config['generation']['output_path']
                    # Check if the directory exists, and if not, create it
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    
                    generator.generate_synthetic_data(
                        #synthesizer=synthesizer,
                        num_sequences=config['generation']['num_sequences'],
                        sequence_length=config['generation']['sequence_length'],
                        output_path=f"{folder_path}/{epoch}.csv",
                        seq_key=config['metadata']['seq_key'],          # Sequence key used to sort by
                        seq_index=config['metadata']['seq_index'],      # Sequence index used to sort by
                    )
                    logging.info(f"Phase 2.4: Synthetic Data Generation with {epoch} epochs Completed.")
        
                
                # Phase 3: Synthetic Data Evaluation
                logging.info(f"Phase 3: Synthetic Data Evaluation Started.")
                
                # Define an evaluator
                real_data_path = real_file
                synth_folder = config['evaluation']['synth_folder']
                exclude_cols = config['evaluation']['exclude_cols']
                sequence_length = config['generation']['sequence_length']
                evaluator = PopulationFidelity(real_data_path, synth_folder, exclude_cols, sequence_length)

                folder_path = config['evaluation']['awd_output']
                # Check if the directory exists, and if not, create it
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
        
                msas_output = config['evaluation']['msas_output']
                #evaluator.compute_msas(output_file=msas_output, x_step=1)
        
                awd_output = config['evaluation']['awd_output']
                evaluator.awd(awd_output)
                
                logging.info(f"Phase 3: Synthetic Data Evaluation Completed.")
            
                # Phase 4: Classification Process
                logging.info(f"Phase 4: Classification Process Started.")
                classifier = Classifier()
        
                real_dataset_path = real_file
                synthetic_folder_path = config['evaluation']['synth_folder']
                seq_index_col = config['classification']['seq_index_col']
                target_col = config['classification']['target_col']
                metric = config['classification']['metric']
                param_grid = config['classification']['param_grid']
                test_size = config['classification']['test_size']
                random_state = config['classification']['random_state']
        
        
                classifier.classify(real_dataset_path=real_dataset_path, 
                                    synthetic_folder_path=synthetic_folder_path,
                                    seq_index_col=seq_index_col, target_col=target_col,
                                    metric=metric, param_grid=param_grid, test_size=test_size, random_state=random_state)
                
                logging.info(f"Phase 4: Classification Process Completed.")
        
                
                # Phase 5: Correlation Analysis
                logging.info(f"Phase 5: Correlation Analysis Started.")
                
                # Load paths from config
                f1_ratios_path = config['correlation_analysis']['f1_ratios_path']
                metric_file_paths = config['correlation_analysis']['metric_file_paths']
                metric_columns = config['correlation_analysis']['metric_columns']  # List of metrics like ["MSAS", "AWD"]
                plot_output_path = config['correlation_analysis']['plot_output_path']
                correlation_output_path = './outputs/correlation_results.csv'  # Path to save the CSV with correlation results
                folder_path = config['correlation_analysis']['plot_output_path']
                # Check if the directory exists, and if not, create it
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
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
