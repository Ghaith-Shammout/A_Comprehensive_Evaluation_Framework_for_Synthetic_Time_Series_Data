def generate_synthetic_data(synthesizer, num_sequences, sequence_length, output_path):
    synthetic_data = synthesizer.sample(num_sequences=num_sequences, sequence_length=sequence_length)
    synthetic_data.to_csv(output_path, index=False)
    print(f"[+] Synthetic data saved to {output_path}")
