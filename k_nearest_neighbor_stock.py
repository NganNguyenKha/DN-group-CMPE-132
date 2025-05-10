import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tabulate import tabulate

#Hide warnings, no issues to actual code
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Paths
cleaned_dir = '../data/cleaned_datasets'
output_dir = '../data/knn_stock_results'
os.makedirs(output_dir, exist_ok=True)

# Get all cleaned files
cleaned_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.csv')]

for file in cleaned_files:
    print(f"\nProcessing file (KNN Stock, 15 features): {file}")

    df = pd.read_csv(os.path.join(cleaned_dir, file))

    if 'Label' not in df.columns:
        print(f"Skipping {file}: No Label column.")
        continue

    X = df.drop('Label', axis=1)
    y = df['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature Selection: top 15 features
    selector = SelectKBest(score_func=f_classif, k=15)
    X_selected = selector.fit_transform(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    knn_model = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1
    )
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    unique_attacks = np.unique(y_test)
    results = []

    for attack in unique_attacks:
        attack_indices = (y_test == attack)
        acc = accuracy_score(y_test[attack_indices], y_pred[attack_indices])
        recall = recall_score(y_test[attack_indices], y_pred[attack_indices], average='macro', zero_division=0)
        precision = precision_score(y_test[attack_indices], y_pred[attack_indices], average='macro', zero_division=0)
        f1 = f1_score(y_test[attack_indices], y_pred[attack_indices], average='macro', zero_division=0)
        results.append((attack, round(acc, 4), round(recall, 4), round(precision, 4), round(f1, 4)))

    results_df = pd.DataFrame(results, columns=['Attack Type', 'Accuracy', 'Recall', 'Precision', 'F1-Score'])
    print(tabulate(results_df, headers='keys', tablefmt='pretty'))
    results_df.to_csv(os.path.join(output_dir, f"knn_stock_{file.replace('.csv', '')}_metrics.csv"), index=False)
