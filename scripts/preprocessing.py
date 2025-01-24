import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

class Preprocessing:
    """
    A class for preprocessing datasets, including removing unwanted columns, 
    enforcing date formats, normalization, label encoding, and sliding window techniques.
    """

    def __init__(self, input_file: str, output_file: str):
        """
        Initialize the DataPreprocessor with input and output file paths.

        Parameters:
        - input_file (str): Path to the input CSV file.
        - output_file (str): Path to save the preprocessed CSV file.
        """
        self.input_file = input_file
        self.output_file = output_file
        self._validate_file()

    def _validate_file(self):
        """Check if the input file exists."""
        if not os.path.isfile(self.input_file):
            raise FileNotFoundError(f"Input file '{self.input_file}' does not exist.")

    def remove_unwanted_columns(self, unwanted_columns: list[str]):
        """
        Remove unwanted columns and rows with null values from the dataset.

        Parameters:
        - unwanted_columns (list[str]): List of column names to remove.
        """
        try:
            df = pd.read_csv(self.input_file)
            # TODO: enable multiple solutions for handling missing values
            df.dropna(inplace=True)
            df.drop(columns=unwanted_columns, inplace=True, errors='ignore')
            df.to_csv(self.output_file, index=False)
            print(f"[+] Removed columns: {unwanted_columns}. Output saved to '{self.output_file}'.")
        except FileNotFoundError:
            print(f"[-] The file {self.input_file} was not found.")
        except KeyError:
            print(f"[-] One or more columns {unwanted_columns} not found in the dataset.")
        except Exception as e:
            print(f"[-] Error while removing unwanted columns: {e}")

    def enforce_date_format(self, date_column: str, date_format: str):
        """
        Ensure all values in the date column follow the specified format and sort by date.

        Parameters:
        - date_column (str): Name of the date column to format.
        - date_format (str): Date format to enforce (e.g., '%Y-%m-%d').
        """
        try:
            df = pd.read_csv(self.output_file)
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
            # TODO: Check if I need to sort by SID then Date
            df.sort_values(by=date_column, ascending=True, inplace=True)
            df[date_column] = df[date_column].dt.strftime(date_format)
            df.to_csv(self.output_file, index=False)
            print(f"[+] Date column '{date_column}' formatted and sorted. Output saved to '{self.output_file}'.")
        except FileNotFoundError:
            print(f"[-] The file {self.output_file} was not found.")
        except ValueError as e:
            print(f"[-] Error parsing dates: {e}")
        except Exception as e:
            print(f"[-] Error while enforcing date format: {e}")

    def normalize_columns(self, method: str, columns: list[str] = None):
        """
        Normalize specified numerical columns using the selected method.

        Parameters:
        - method (str): Normalization method ('minmax' or 'zscore').
        - columns (list[str], optional): Columns to normalize. If None, all numerical columns are used.
        """
        try:
            df = pd.read_csv(self.output_file)
            columns = columns or df.select_dtypes(include='number').columns.tolist()
            scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
            df.to_csv(self.output_file, index=False)
            print(f"[+] Normalized columns {columns} using {method} method. Output saved to '{self.output_file}'.")
        except FileNotFoundError:
            print(f"[-] The file {self.output_file} was not found.")
        except ValueError as e:
            print(f"[-] Error: {e}")
        except Exception as e:
            print(f"[-] Error while normalizing data: {e}")

    def label_encode_columns(self, categorical_columns: list[str]):
        """
        Label encode specified categorical columns.

        Parameters:
        - categorical_columns (list[str]): List of categorical column names to encode.
        """
        try:
            df = pd.read_csv(self.output_file)
            label_encoders = {}
            for col in categorical_columns:
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])
                label_encoders[col] = label_encoder
            df.to_csv(self.output_file, index=False)
            print(f"[+] Label encoded columns: {categorical_columns}. Output saved to '{self.output_file}'.")
            return label_encoders
        except FileNotFoundError:
            print(f"[-] The file {self.output_file} was not found.")
        except KeyError as e:
            print(f"[-] Error encoding column {e}: Column not found.")
        except Exception as e:
            print(f"[-] Error during label encoding: {e}")

    def apply_sliding_window(self, window_size: int, step_size: int):
        """
        Apply sliding window technique to the dataset.
    
        Parameters:
        - window_size (int): Size of the window.
        - step_size (int): Step size for moving the window.
    
        Returns:
        - int: Number of generated windows.
        """
        df = pd.read_csv(self.output_file)
        if len(df) < window_size:
            raise ValueError("Window size is larger than the dataset.")
        windows = [
            df.iloc[i:i + window_size].assign(SID=i // step_size + 1)
            for i in range(0, len(df) - window_size + 1, step_size)
        ]
        result = pd.concat(windows, axis=0)
        result.to_csv(self.output_file, index=False)
        print(f"[+] Applied sliding window. Generated {len(windows)} windows. Output saved to '{self.output_file}'.")
        return len(windows)

            
   