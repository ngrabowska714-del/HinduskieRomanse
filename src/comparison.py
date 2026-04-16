"""
Szkielet dla Osoby 4.
Cel: porównanie własnej sieci z benchmarkiem ze sklearn.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score

def run_sklearn_comparison(task_name: str) -> dict[str, Any]:
    """
    Implementacja Osoby 4
    Porównanie wykorzystujące bazowe MLPClassifier / MLPRegressor z biblioteki sklearn.
    """
    # Wczytanie danych z odpowiego pliku .npz
    data_path = f"data/processed/{task_name}.npz"
    data = np.load(data_path)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # Redukcja wymiaru y dla regresji celem usunięcia ostrzeżeń biblioteki
    y_train_flat = y_train.ravel()
    y_test_flat = y_test.ravel()

    results = {}

    if "classification" in task_name:
        model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=200, random_state=42)
        model.fit(X_train, y_train_flat)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train_flat, train_pred)
        test_acc = accuracy_score(y_test_flat, test_pred)
        
        results = {
            "model": "MLPClassifier",
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
        }
    elif "regression" in task_name:
        model = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=200, random_state=42)
        model.fit(X_train, y_train_flat)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train_flat, train_pred)
        test_r2 = r2_score(y_test_flat, test_pred)
        
        results = {
            "model": "MLPRegressor",
            "train_r2": train_r2,
            "test_r2": test_r2
        }
    else:
        raise ValueError(f"Nieznany typ zadania: {task_name}")

    print(f"Wyniki sklearn dla zadania {task_name}: {results}")
    return results

if __name__ == "__main__":
    # Szybki test działania
    run_sklearn_comparison("classification_divorce")
    run_sklearn_comparison("regression_children")
