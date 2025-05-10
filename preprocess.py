import os
import pandas as pd
import numpy as np

# Paths
dataset_dir = '../data'
output_dir = '../data/cleaned_datasets'
os.makedirs(output_dir, exist_ok=True)

# List of datasets
datasets = [
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX',
    'Friday-WorkingHours-Morning.pcap_ISCX',
    'Monday-WorkingHours.pcap_ISCX',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX',
    'Tuesday-WorkingHours.pcap_ISCX',
    'Wednesday-workingHours.pcap_ISCX'
]

def preprocess_dataset(filename):
    raw_path = os.path.join(dataset_dir, filename + '.csv')
    print(f"\n Preprocessing: {filename}")

    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        print(f" Failed to read {filename}: {e}")
        return None

    df.columns = df.columns.str.strip()
    df.drop(columns=[col for col in ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'] if col in df.columns], errors='ignore', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.round(6)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    if 'Label' not in df.columns:
        print(f" Label column missing in {filename}, skipping.")
        return None

    # Note: Do not encode Label column here (keep attack names)

    cleaned_path = os.path.join(output_dir, f'cleaned_{filename.lower().replace(".", "_")}.csv')
    df.to_csv(cleaned_path, index=False)
    print(f" Saved cleaned file: {cleaned_path}")
    return cleaned_path

# Main preprocess loop
if __name__ == "__main__":
    for dataset in datasets:
        preprocess_dataset(dataset)

