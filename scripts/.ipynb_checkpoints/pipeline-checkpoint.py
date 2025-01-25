# File: pipeline.py
import logging
from pathlib import Path
import yaml
from preprocessing import Preprocessing
from generating import SyntheticDataGenerator
from population_fidelity import PopulationFidelity
from classification import Classifier
from correlation_analysis import CorrelationAnalysis


def setup_logging(log_file: str, level: str) -> None:
    """Configures logging for the pipeline."""
    logging.basicConfig(
        filename=log_file,
        level=level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_config(config_path: str) -> dict:
    """Loads the pipeline configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensures that a directory exists. Creates the directory tree if it doesn't exist.
    """
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {directory_path}")
    else:
        logging.info(f"Directory already exists: {directory_path}")


def process_config_file(config_file: Path) -> None:
    """Processes a single configuration file."""
    config = load_config(config_file)
    setup_logging(config["logging"]["log_file"], config["logging"]["level"])
    logging.info(f"Pipeline execution for {config_file.name} started.")

    try:
        # Call phase functions
        source_file = config["folders"]["source_file"]
        source_name = Path(source_file).stem
        data_folder = f"./outputs/{source_name}/data"
        ensure_directory_exists(data_folder)
        real_file = f"{data_folder}/Real_{source_name}{Path(source_file).suffix}"

        # Execute pipeline phases
        #seq_num = preprocessing(config, source_file, real_file)
        #synthetic_data_generation(config, real_file, source_name, seq_num)
        #evaluation(config, real_file, source_name)
        #classification(config, real_file, source_name)
        correlation_analysis(config, source_name)

        logging.info(f"Pipeline execution for {config_file.name} completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline execution failed for {config_file.name}: {e}")
        raise


def preprocessing(config: dict, source_file: str, real_file: str) -> int:
    """Executes the data preprocessing phase."""
    logging.info("Phase 1: Data Preprocessing Started.")
    preprocessor = Preprocessing(input_file=source_file, output_file=real_file)
    preprocessor.remove_unwanted_columns(config["preprocessing"]["unwanted_columns"])
    preprocessor.enforce_date_format(
        date_column=config["dataset"]["date_col"],
        date_format=config["dataset"]["date_format"],
    )
    preprocessor.normalize_columns(
        method=config["preprocessing"]["normalization_method"],
        columns=config["dataset"]["normalize_columns"],
    )
    preprocessor.label_encode_columns(
        categorical_columns=config["preprocessing"]["categorical_columns"]
    )
    seq_num = preprocessor.apply_sliding_window(
        window_size=config["preprocessing"]["window_size"],
        step_size=config["preprocessing"]["step_size"],
    )
    logging.info("Phase 1 Completed Successfully.")
    return seq_num

def synthetic_data_generation(config: dict, real_file: str, source_name: str, seq_num: int) -> None:
    """Executes the synthetic data generation phase."""
    logging.info("Phase 2: Synthetic Data Generation Started.")
    generator = SyntheticDataGenerator()
    metadata_dir = f"./outputs/{source_name}/data/metadata_{source_name}.json"
    generator.define_metadata(
        dataset_path=real_file,
        sequence_key=config["metadata"]["seq_key"],
        sequence_index=config["metadata"]["seq_index"],
        date_format=config["dataset"]["date_format"],
        metadata_path=metadata_dir,
    )
    logging.info("Metadata Definition Completed.")
    
    for epoch in config['synthesizer']['epochs']:
        logging.info(f"Phase 2.2: Synthesizer Initializing with {epoch} epochs Started.")
        models_dir = f"./outputs/{source_name}/Models/"
        synthesizer = generator.load_synthesizer(models_dir, epoch)
        if not synthesizer:
            synthesizer = generator.initialize_synthesizer(
                metadata_path=metadata_dir,
                context_columns=config['synthesizer']['context_columns'],
                epochs=epoch,
                verbose=config['synthesizer']['verbose'],
                cuda=config['synthesizer']['cuda']
            )
            logging.info(f"Phase 2.2: Synthesizer Initializing with {epoch} epochs Successfully Completed.")
            
            logging.info(f"Phase 2.3: Synthesizer Training with {epoch} epochs Started.")
            train_folder_path = f"./outputs/{source_name}/Models"
            ensure_directory_exists(train_folder_path)
            generator.train_synthesizer(
                train_dataset_path=real_file,          
                model_save_path = f"{train_folder_path}/{epoch}.pkl"
            )
            logging.info(f"Phase 2.3: Synthesizer Training with {epoch} epochs Completed.")
            
        # Phase 2.4: Synthetic Data Generation
        logging.info(f"Phase 2.4: Synthetic Data Generation with {epoch} epochs Started.")
        synth_folder_path = f"./outputs/{source_name}/Synth_Data/{epoch}"
        ensure_directory_exists(synth_folder_path)
       
        generator.generate_synthetic_data(
            num_sequences = seq_num,
            sequence_length = config['preprocessing']['window_size'],
            output_dir = synth_folder_path,
            sequence_key = config['metadata']['seq_key'],
            sequence_index = config['metadata']['seq_index'],
            num_files = 5
        )
        logging.info(f"Phase 2.4: Synthetic Data Generation with {epoch} epochs Completed.")

def evaluation(config: dict, real_file: str, source_name: str) -> None:
    """Executes the synthetic data evaluation phase."""
    logging.info("Phase 3: Synthetic Data Evaluation Started.")
    
    # Define paths
    synth_folder_path = Path(f"./outputs/{source_name}/Synth_Data")
    eva_folder_path = Path(f"./outputs/{source_name}/Evaluation")
    plot_output_path = Path(f"./outputs/{source_name}/Plots")
    
    # Ensure directories exist
    ensure_directory_exists(eva_folder_path)
    ensure_directory_exists(plot_output_path)
    
    # Check if the synthetic data folder exists
    if not synth_folder_path.exists():
        logging.error(f"Synthetic data folder not found: {synth_folder_path}")
        return
    
    # Iterate through every folder inside synth_folder_path
    for folder in synth_folder_path.iterdir():
        if folder.is_dir():
            logging.info(f"[+] Evaluating folder: {folder}")
            try:
                # Initialize the evaluator
                evaluator = PopulationFidelity(real_file, folder, eva_folder_path)

                # Compute MSAS
                exclude_cols = config['evaluation']['exclude_cols']
                evaluator.compute_msas(x_step=1, exclude_columns=exclude_cols)
            
                # Compute AWD
                evaluator.compute_awd(exclude_columns=exclude_cols, plot_output_path=plot_output_path)
            
                # Compute TC
                SID = config['dataset']['SID']
                numerical_col = config['dataset']['normalize_columns']
                top_peaks = config['evaluation']['top_peaks']
                evaluator.compute_temporal_correlation(sequence_id=SID, channel_columns=numerical_col, top_peaks=top_peaks)
            
            except Exception as e:
                logging.error(f"Error evaluating folder {folder}: {e}")
    
    logging.info("Phase 3: Synthetic Data Evaluation Completed.")


def classification(config: dict, real_file: str, source_name: str) -> None:
  """Executes the classification phase."""
  logging.info("Phase 4: Classification Process Started.")
  classifier = Classifier()

  synth_folder_path = Path(f"./outputs/{source_name}/Synth_Data")
  seq_index_col = config['dataset']['SID']
  target_col = config['classification']['target_col']
  param_grid = config['classification']['param_grid']
  test_size = config['classification']['test_size']
  random_state = config['classification']['random_state']
  metric_output_file = f"./outputs/{source_name}/Evaluation"

  # Ensure directory exists
  ensure_directory_exists(metric_output_file)
  ensure_directory_exists(metric_output_file)

  # Iterate through every folder inside synth_folder_path
  for folder in synth_folder_path.iterdir():
    if folder.is_dir():
      logging.info(f"[+] Evaluating folder: {folder}")
      try:
        classifier.compute(real_dataset_path=real_file, synthetic_folder_path=folder, seq_index_col=seq_index_col,
                           target_col=target_col, param_grid=param_grid, test_size=test_size, random_state=random_state,
                           output_dir=metric_output_file)
      except Exception as e:
        logging.error(f"Error evaluating folder {folder}: {e}")
            
  logging.info("Phase 4: Classification Process Completed.")
        
def correlation_analysis(config: dict, source_name: str) -> None:
  """Executes the correlation analysis phase."""
  logging.info("Phase 5: Correlation Analysis Started.")
  evaluation_path = f"./outputs/{source_name}/Evaluation"
  metric_columns = ["MSAS", "AWD", "TC"]
  plot_output_path = f"./outputs/{source_name}/Plots"
  
  analysis = CorrelationAnalysis()
  
  # Example usage
  folders = ['MSAS', 'AWD', 'TC', 'f1']
  
  # Call the method
  analysis.calculate_averages_and_save(evaluation_path, folders)
  
  analysis.compute_correlation( 
    evaluation_path = evaluation_path, 
    metric_columns = metric_columns, 
    plot_path=plot_output_path)

  logging.info("Phase 5: Correlation Analysis Completed.")



def main() -> None:
    """Main pipeline function."""
    config_dir = Path("config/")
    for config_file in config_dir.glob("*.yaml"):
        logging.info(f"Processing configuration file: {config_file.name}")
        process_config_file(config_file)


if __name__ == "__main__":
    main()
