"""
Szkielet dla Osoby 2.
Cel: własna implementacja SSN w NumPy, zgodna z danymi z `src.data_loader`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


class NeuralNetwork:
    def __init__(
        self,
        layer_sizes: list[int],
        problem_type: str,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        weight_init: str = "xavier",
        random_state: int = 42,
    ) -> None:
        self.layer_sizes = layer_sizes
        self.problem_type = problem_type
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_init = weight_init
        self.random_state = random_state

        # TODO: Osoba 2 -> inicjalizacja wag i biasów
        self.weights = []
        self.biases = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainingHistory:
        """
        TODO: Osoba 2
        Oczekiwane minimum:
        - forward pass
        - backward pass
        - update wag
        - zapis historii loss
        """
        raise NotImplementedError("To implementuje Osoba 2.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        TODO: Osoba 2
        - regresja: zwróć wartości ciągłe
        - klasyfikacja binarna: zwróć 0/1 albo probability + osobna metoda predict_proba
        """
        raise NotImplementedError("To implementuje Osoba 2.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Opcjonalne, ale przydatne dla klasyfikacji.
        """
        raise NotImplementedError("To implementuje Osoba 2.")
