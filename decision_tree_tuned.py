import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tabulate import tabulate

#Hide warnings, no issues to actual code
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Paths
cleaned_dir = '../data/cleaned_datasets'
output_dir = '../data/dt_tuned_results'
os.makedirs(output_dir, exist_ok=True)

# List of cleaned files
cleaned_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.csv')]

# Process each file
for file in cleaned_files:
    print(f"\nProcessing file (Decision Tree Tuned, 40 features): {file}")

    df = pd.read_csv(os.path.join(cleaned_dir, file))

    if 'Label' not in df.columns:
        print(f"Skipping {file}: No Label column.")
        continue

    X = df.drop('Label', axis=1)
    y = df['Label']

    # Feature selection: top 40 using ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k=40)
    X_selected = selector.fit_transform(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # Skip if only one class
    if len(np.unique(y_train)) < 2:
       print(f"Skipping {file}: Only one class in training set.")
       continue

    # Tuned Decision Tree
    dt_model = DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=4,
        class_weight='balanced',
        random_state=42
    )
    dt_model.fit(X_train, y_train)

    y_pred = dt_model.predict(X_test)

    # Per-class evaluation
    unique_attacks = np.unique(y_test)
    results = []

    for attack in unique_attacks:
        attack_indices = (y_test == attack)
        acc = accuracy_score(y_test[attack_indices], y_pred[attack_indices])
        recall = recall_score(y_test[attack_indices], y_pred[attack_indices], average='macro', zero_division=0)
        precision = precision_score(y_test[attack_indices], y_pred[attack_indices], average='macro', zero_division=0)
        f1 = f1_score(y_test[attack_indices], y_pred[attack_indices], average='macro', zero_division=0)
        results.append((attack, round(acc, 4), round(recall, 4), round(precision, 4), round(f1, 4)))

    # Output results
    results_df = pd.DataFrame(results, columns=['Attack Type', 'Accuracy', 'Recall', 'Precision', 'F1-Score'])
    print(tabulate(results_df, headers='keys', tablefmt='pretty'))

    # Save results
    results_df.to_csv(os.path.join(output_dir, f"dt_tuned_{file.replace('.csv', '')}_metrics.csv"), index=False)

