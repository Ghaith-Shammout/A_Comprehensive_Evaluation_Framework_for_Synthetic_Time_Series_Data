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

def ensure_directory_exists(directory_path):
    """
    Checks if the given directory exists. If it doesn't, creates the entire directory tree.
    
    Parameters:
    - directory_path (str): Path of the directory to check and create if necessary.
    """
    try:
        # Check if the directory exists
        if not os.path.exists(directory_path):
            # Create the entire directory tree
            os.makedirs(directory_path, exist_ok=True)
            print(f"[+] Directory '{directory_path}' created successfully.")
        else:
            print(f"[+] Directory '{directory_path}' already exists.")
    except PermissionError:
        print(f"[-] Permission denied when accessing or creating directory '{directory_path}'.")
    except Exception as e:
        print(f"[-] Unexpected error occurred: {e}")



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
            logging.info(f"Pipeline execution for {filename} started.")   

            try:
                # Loading source_file from the configuration
                source_file = config['folders']['source_file']
                
                # Extract the base name without the extension, e.g. Steal.csv -> Steal
                base_name = os.path.splitext(os.path.basename(source_file))[0]

                # Define Data folder 
                data_folder = f"./outputs/{base_name}/data"
                
                # Ensure Directory Exists
                ensure_directory_exists(data_folder)
                
                # Add the prefix "Real_" and reattach the extension
                real_file = f"{data_folder}/Real_{base_name}{os.path.splitext(source_file)[1]}"
                """
                # Phase 1: Data Preprocessing
                logging.info("Phase 1: Data Preprocessing Started.")
                # Create an instance of the Preprocessing class
                preprocessor = Preprocessing(
                    input_file=source_file,
                    output_file=real_file,
                )
                
                # running preprocessing steps with their respective arguments
                preprocessor.remove_unwanted_columns(unwanted_columns=config['preprocessing']['unwanted_columns'])
                preprocessor.enforce_date_format(date_column=config['dataset']['date_col'],
                                                 date_format=config['dataset']['date_format'])
                preprocessor.normalize_data(method=config['preprocessing']['normalization_method'], 
                                           normalize_columns=config['dataset']['numerical_col'])
                preprocessor.label_encode_and_save(categorical_columns=config['preprocessing']['categorical_columns'])
                num_seq = preprocessor.sliding_window(window_size=config['preprocessing']['window_size'],
                                                      step_size=config['preprocessing']['step_size'])
                logging.info("Phase 1 Completed Successfully.")

                # Phase 2: Synthetic Data Generation 
                logging.info("Phase 2: Synthetic Data Generation Process Started.")
                
                # Define a generator
                generator = SyntheticDataGenerator()
                
                logging.info("Phase 2.1: Metadata Definition started.")

                # Define directory to store metadata file
                metadata_dir = f"./outputs/{base_name}/data/"
                
                # Ensure the directory exists
                ensure_directory_exists(metadata_dir)
                
                # Construct the metadata filename using the metadata_path
                metadata_filename = os.path.join(metadata_dir, f"metadata_{base_name}.json")

                # Define metadata
                metadata = generator.define_metadata(
                    dataset=real_file,
                    seq_key=config['metadata']['seq_key'],  # Sequence key
                    seq_index=config['metadata']['seq_index'],      # Sequence index
                    date_format=config['dataset']['date_format'],
                    metadata_path=metadata_filename
                )
                logging.info("Phase 2.1: Metadata Definition Completed Successfully.")

                
                for epoch in config['synthesizer']['epochs']:
                    logging.info(f"Starting pipeline run with {epoch} epochs.")
        
                    # Update the number of epochs in the configuration dynamically
                    logging.info(f"Phase 2.2: Synthesizer Initializing with {epoch} epochs Started.")
                    # Train the synthesizer
                    synthesizer = generator.initialize_synthesizer(
                        metadata_path=metadata_filename,
                        context_columns=config['synthesizer']['context_columns'],
                        epochs=epoch,
                        verbose=config['synthesizer']['verbose'],
                        cuda=config['synthesizer']['cuda']
                    )
                    logging.info(f"Phase 2.2: Synthesizer Initializing with {epoch} epochs Successfully Completed.")
        
                    # Phase 2.3: Synthesizer Training
                    logging.info(f"Phase 2.3: Synthesizer Training with {epoch} epochs Started.")

                    train_folder_path = f"./outputs/{base_name}/Models"

                    # Ensure directory exists
                    ensure_directory_exists(train_folder_path)
                    
                    generator.train_synthesizer(
                        #synthesizer=synthesizer,
                        train_dataset=real_file,  # Ensure DataFrame is passed                    
                        save_path = f"{train_folder_path}/{epoch}.pkl"
                    )
                    logging.info(f"Phase 2.3: Synthesizer Training with {epoch} epochs Completed.")
        
                    # Phase 2.4: Synthetic Data Generation
                    logging.info(f"Phase 2.4: Synthetic Data Generation with {epoch} epochs Started.")
                    logging.info("Phase 4: Synthetic Data Generation started.")
                    
                    synth_folder_path = f"./outputs/{base_name}/Synth_Data"

                    # Ensure directory exists
                    ensure_directory_exists(synth_folder_path)
                    
                    generator.generate_synthetic_data(
                        #synthesizer=synthesizer,
                        num_sequences=num_seq,
                        sequence_length=config['preprocessing']['window_size'],
                        output_path=f"{synth_folder_path}/{epoch}.csv",
                        seq_key=config['metadata']['seq_key'],          # Sequence key used to sort by
                        seq_index=config['metadata']['seq_index'],      # Sequence index used to sort by
                    )
                    logging.info(f"Phase 2.4: Synthetic Data Generation with {epoch} epochs Completed.")
                

                # TODO: Remove when uncommenting Phase 2
                synth_folder_path = f"./outputs/{base_name}/Synth_Data"
                
                # Phase 3: Synthetic Data Evaluation
                logging.info(f"Phase 3: Synthetic Data Evaluation Started.")
                
                # Define parameters for the evaluator
                exclude_cols = config['evaluation']['exclude_cols']

                # Define evaluation folder path
                eva_folder_path = f"./outputs/{base_name}/Evaluation"
                # Ensure the directory exists
                ensure_directory_exists(eva_folder_path)

                plot_output_path = f"./outputs/{base_name}/Plots/"
                ensure_directory_exists(plot_output_path)

                # Initialize the evaluator 
                evaluator = PopulationFidelity(real_file, synth_folder_path, eva_folder_path)
                
                # Compute MSAS
                evaluator.compute_msas(x_step=1, exclude_columns=exclude_cols)
                # Compute AWD
                evaluator.compute_awd(exclude_columns=exclude_cols, plot_output_path=plot_output_path)
                # Compute TC
                SID = config['dataset']['SID']
                numerical_col = config['dataset']['numerical_col']
                top_peaks = config['evaluation']['top_peaks']
                evaluator.compute_temp_corr(seq_id=SID,
                                            channel_cols=numerical_col,
                                            top_peaks=top_peaks)
                
                logging.info(f"Phase 3: Synthetic Data Evaluation Completed.")
                """
                # TODO: Remove when uncommenting Phase 2
                synth_folder_path = f"./outputs/{base_name}/Synth_Data"
                eva_folder_path = f"./outputs/{base_name}/Evaluation"
                plot_output_path = f"./outputs/{base_name}/Plots/"
                
                # Phase 4: Classification Process
                logging.info(f"Phase 4: Classification Process Started.")
                classifier = Classifier()
        
                real_dataset_path = real_file
                synthetic_folder_path = synth_folder_path
                seq_index_col = config['dataset']['SID']
                target_col = config['classification']['target_col']
                param_grid = config['classification']['param_grid']
                test_size = config['classification']['test_size']
                random_state = config['classification']['random_state']
                metric_output_file = eva_folder_path

                # Ensure directory exists
                ensure_directory_exists(metric_output_file)
        
                classifier.classify(real_dataset_path=real_dataset_path, 
                                    synthetic_folder_path=synthetic_folder_path,
                                    seq_index_col=seq_index_col,
                                    target_col=target_col,
                                    param_grid=param_grid,
                                    test_size=test_size,
                                    random_state=random_state,
                                    metric_output_file=metric_output_file)
                
                logging.info(f"Phase 4: Classification Process Completed.")
        
                # Phase 5: Correlation Analysis
                logging.info(f"Phase 5: Correlation Analysis Started.")
                
                # Load paths from config
                f1_ratios_path =  f"./outputs/{base_name}/Evaluation/f1_ratios.csv"
                metric_columns = config['correlation_analysis']['metric_columns']  # List of metrics like ["MSAS", "AWD"]
               
                correlation_output_path = eva_folder_path
                
                # Check if the directory exists, and if not, create it
                #ensure_directory_exists(folder_path)
                
                # Instantiate the CorrelationAnalysis class
                analysis = CorrelationAnalysis()
                
                # Perform correlation analysis for all metrics and save results
                analysis.correlation_analysis(
                    f1_ratios_path = f1_ratios_path,
                    metric_columns=metric_columns,
                    base_name=base_name,
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
