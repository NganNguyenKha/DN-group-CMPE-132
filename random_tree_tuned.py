import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tabulate import tabulate

#Hide warnings, no issues to actual code
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Paths
cleaned_dir = '../data/cleaned_datasets'
output_dir = '../data/rf_tuned_results'
os.makedirs(output_dir, exist_ok=True)

# Get all cleaned files
cleaned_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.csv')]

for file in cleaned_files:
    print(f"\nProcessing file (Tuned RF): {file}")

    df = pd.read_csv(os.path.join(cleaned_dir, file))

    if 'Label' not in df.columns:
        print(f"Skipping {file}: No Label column.")
        continue

    X = df.drop('Label', axis=1)
    y = df['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature Selection: Keep top 40 features
    selector = SelectKBest(score_func=f_classif, k=40)
    X_selected = selector.fit_transform(X_scaled, y)

    # Optional: debug/print selected features
    # selected_features = X.columns[selector.get_support()]
    # print(f"Selected features: {list(selected_features)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    unique_attacks = np.unique(y_test)
    results = []

    for attack in unique_attacks:
        attack_indices = (y_test == attack)
        attack_y_true = y_test[attack_indices]
        attack_y_pred = y_pred[attack_indices]

        acc = accuracy_score(attack_y_true, attack_y_pred)
        recall = recall_score(attack_y_true, attack_y_pred, average='macro', zero_division=0)
        precision = precision_score(attack_y_true, attack_y_pred, average='macro', zero_division=0)
        f1 = f1_score(attack_y_true, attack_y_pred, average='macro', zero_division=0)

        results.append((attack, round(acc, 4), round(recall, 4), round(precision, 4), round(f1, 4)))

    results_df = pd.DataFrame(results, columns=['Attack Type', 'Accuracy', 'Recall', 'Precision', 'F1-Score'])
    print(tabulate(results_df, headers='keys', tablefmt='pretty'))
    results_df.to_csv(os.path.join(output_dir, f"rf_tuned_{file.replace('.csv', '')}_metrics.csv"), index=False)
