from sdv.sequential import PARSynthesizer

synthesizer = PARSynthesizer.load(filepath='2500.pkl')
synthesizer._model._model.rnn.flatten_parameters()

for i in range(1, 10 + 1):
    # Generate synthetic data
    synthetic_data = synthesizer.sample(num_sequences=2917, sequence_length=48)
    
    # Sort the data by the provided columns
    synthetic_data = synthetic_data.sort_values(by=['SID', 'date'], ascending=True)
    
    # Generate dynamic file name (e.g., v1.csv, v2.csv, etc.)
    output_path = f"{i}.csv"
    
    # Save the sorted data to a CSV file
    synthetic_data.to_csv(output_path, index=False)
    
    # Print a message with the column names used for sorting
    print(f"[+] Synthetic file {output_path} saved")