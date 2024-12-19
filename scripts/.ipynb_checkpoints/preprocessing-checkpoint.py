import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

class Preprocessing:
    def __init__(self, input_file, output_file):
        """
        Initialize the Preprocessing class with necessary parameters.
        
        Parameters:
        - input_file (str): Path to the input CSV file.
        - output_file (str): Path to save the preprocessed CSV file.
        """
        self.input_file = input_file
        self.output_file = output_file

    def remove_unwanted_columns(self, unwanted_columns):
        """Remove unwanted columns from the dataset."""
        try:
            # Check if the output directory exists
            output_dir = os.path.dirname(self.output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)  # Create the directory if it doesn't exist
            
            # Read the CSV file
            df = pd.read_csv(self.input_file)
            
            # Drop the unwanted columns
            df.drop(columns=unwanted_columns, inplace=True)
            
            # Save the result to a new CSV file
            df.to_csv(self.output_file, index=False)
            
            # Print the result
            print(f"[+] Columns {unwanted_columns} removed. Output saved to {self.output_file}.")
        except FileNotFoundError:
            print(f"[-] The file {self.input_file} was not found.")
        except KeyError:
            print(f"[-] One or more columns {unwanted_columns} not found in the dataset.")
        except Exception as e:
            print(f"[-] Error while removing unwanted columns: {e}")

    def enforce_date_format(self, date_column, date_format="%d/%m/%Y %H:%M"):
        """Ensure all values in the date column follow the specified format and sort by date."""
        try:
            # Read the dataset from the file
            df = pd.read_csv(self.output_file)
            
            # Check if the date column exists in the DataFrame
            if date_column not in df.columns:
                raise ValueError(f"The {date_column} column is not found in the dataset.")
            
            # Convert the date column to datetime format
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
            
            # Sort the DataFrame by the date column in ascending order (use `ascending=False` for descending order)
            df = df.sort_values(by=date_column, ascending=True)
            
            # Format the date column back to the specified string format
            df[date_column] = df[date_column].dt.strftime(date_format)
            
            # Save the updated DataFrame to the output file
            df.to_csv(self.output_file, index=False)
            print(f"[+] Date format standardized and sorted by {date_column}, then saved to '{self.output_file}'.")
        
        except FileNotFoundError:
            print(f"[-] The file {self.output_file} was not found.")
        except ValueError as e:
            print(f"[-] Error parsing dates: {e}")
        except Exception as e:
            print(f"[-] Error while enforcing date format: {e}")

    def normalize_data(self, method='minmax', normalize_columns=None):
        """Normalize specified numerical columns using the specified method."""
        try:
            # Read the dataset from the file
            df = pd.read_csv(self.output_file)
            
            # If normalize_columns is provided, use it; otherwise, find numerical columns automatically
            if normalize_columns is not None:
                # Ensure the specified columns exist in the DataFrame
                missing_cols = [col for col in normalize_columns if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"[-] The following columns were not found in the dataset: {', '.join(missing_cols)}")
                numerical_cols = normalize_columns
            else:
                # Dynamically select numerical columns (columns with numeric types like int or float)
                numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
            # Check if there are any numerical columns to normalize
            if not numerical_cols:
                raise ValueError("[-] No numerical columns found for normalization.")
            
            # Select the normalization method
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'zscore':
                scaler = StandardScaler()
            else:
                raise ValueError("[-] Unsupported normalization method. Use 'minmax' or 'zscore'.")
    
            # Apply normalization
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            
            # Save the normalized data to the output file
            df.to_csv(self.output_file, index=False)
            print(f"[+] Normalized data saved to '{self.output_file}'.")
        
        except FileNotFoundError:
            print(f"[-] The file {self.output_file} was not found.")
        except ValueError as e:
            print(f"[-] Error: {e}")
        except Exception as e:
            print(f"[-] Error while normalizing data: {e}")

        

    def label_encode_and_save(self, categorical_columns):
        """Label encode categorical columns and save the updated dataset."""
        try:
            df = pd.read_csv(self.output_file)

            label_encoders = {}
            for column in categorical_columns:
                if column not in df.columns:
                    print(f"[-] Column {column} not found in dataset.")
                    continue

                label_encoder = LabelEncoder()
                df[column] = label_encoder.fit_transform(df[column])
                label_encoders[column] = label_encoder
                '''
                print(f"[+] Encoding mapping for '{column}': {dict(zip(label_encoder.classes_,
                                                                       label_encoder.transform(label_encoder.classes_)
                                                                      )
                                                                  )
                                                            }"
                     )
                '''
            
            df.to_csv(self.output_file, index=False)
            print(f"[+] Updated DataFrame saved to {self.output_file}")
            return df, label_encoders
        except FileNotFoundError:
            print(f"[-] The file {self.output_file} was not found.")
        except KeyError as e:
            print(f"[-] Error encoding column {e}: Column not found.")
        except Exception as e:
            print(f"[-] Error during label encoding: {e}")

    def sliding_window(self, window_size, step_size):
        """Apply sliding window technique to the dataset."""
        try:
            df = pd.read_csv(self.output_file)

            if len(df) < window_size:
                raise ValueError("The window size is larger than the dataset. Adjust the window size.")
            
            windowed_data = []
            sid = 1  # Start the sequence identifier from 1
            
            for start in range(0, len(df) - window_size + 1, step_size):
                end = start + window_size
                window = df.iloc[start:end].copy()
                window['SID'] = sid
                windowed_data.append(window)
                sid += 1

            print(f"[+] Sliding window successfully performed, with number of windows {len(windowed_data)}")
            slided_data = pd.concat(windowed_data, axis=0)
            slided_data.to_csv(self.output_file, index=False)
            print(f"[+] Sliding window data saved to '{self.output_file}'.")
        except FileNotFoundError:
            print(f"[-] The file {self.output_file} was not found.")
        except ValueError as e:
            print(f"[-] Error with window size: {e}")
        except Exception as e:
            print(f"[-] Error while applying sliding window: {e}")

    def preprocess(self, unwanted_columns, date_column, categorical_columns, window_size, step_size, normalization_method='minmax'):
        """Run all preprocessing steps in sequence."""
        try:
            self.remove_unwanted_columns(unwanted_columns)
            self.enforce_date_format(date_column)
            self.normalize_data(method=normalization_method)
            self.label_encode_and_save(categorical_columns)
            self.sliding_window(window_size=window_size, step_size=step_size)
            print(f"[+] Final preprocessed data saved to {self.output_file}.")
        except Exception as e:
            print(f"[-] Error during preprocessing: {e}")
