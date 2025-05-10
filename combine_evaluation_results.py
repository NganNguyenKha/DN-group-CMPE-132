import pandas as pd
import os
from tabulate import tabulate

# Folder mappings (easy to extend later for SVM, KNN)
folders_to_check = {
    'Random Forest Stock': '../data/rf_stock_results',
    'Random Forest Tuned': '../data/rf_tuned_results',
    'Decision Tree Stock': '../data/dt_stock_results',
    'Decision Tree Tuned': '../data/dt_tuned_results',
    'KNN Stock': '../data/knn_stock_results',
    'KNN Tuned': '../data/knn_tuned_results'
}

# Helper function to combine and print results for a given model
def process_results(model_name, results_dir):
    if not os.path.exists(results_dir):
        print(f"\nDirectory {results_dir} does not exist. Skipping {model_name}.")
        return

    result_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]

    if not result_files:
        print(f"\nNo evaluation files found in {results_dir}. Skipping {model_name}.")
        return

    all_results = []
    for file in result_files:
        df = pd.read_csv(os.path.join(results_dir, file))
        all_results.append(df)

    combined_df = pd.concat(all_results, ignore_index=True)

    # Separate BENIGN and other attacks
    benign_df = combined_df[combined_df['Attack Type'] == 'BENIGN']
    attacks_df = combined_df[combined_df['Attack Type'] != 'BENIGN']

    # Calculate mean for BENIGN
    if not benign_df.empty:
        benign_avg = benign_df[['Accuracy', 'Recall', 'Precision', 'F1-Score']].mean()
        benign_row = pd.DataFrame({
            'Attack Type': ['BENIGN'],
            'Accuracy': [round(benign_avg['Accuracy'], 4)],
            'Recall': [round(benign_avg['Recall'], 4)],
            'Precision': [round(benign_avg['Precision'], 4)],
            'F1-Score': [round(benign_avg['F1-Score'], 4)]
        })
    else:
        benign_row = pd.DataFrame()

    # Final combined
    final_combined_df = pd.concat([benign_row, attacks_df], ignore_index=True)
    final_combined_df = final_combined_df[['Attack Type', 'Accuracy', 'Recall', 'Precision', 'F1-Score']]
    final_combined_df = final_combined_df.sort_values('Attack Type').reset_index(drop=True)

    # Print
    print(f"\nCombined Evaluation for {model_name}:\n")
    print(tabulate(final_combined_df, headers='keys', tablefmt='pretty'))

    # (Optional) Save if you want
    # final_combined_df.to_csv(f'../data/{model_name.replace(' ', '_').lower()}_combined.csv', index=False)

# 🚀 Main
if __name__ == "__main__":
    for model_name, results_dir in folders_to_check.items():
        process_results(model_name, results_dir)

