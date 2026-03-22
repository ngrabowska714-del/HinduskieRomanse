"""
Szkielet dla Osoby 3.
Cel: powtarzalne eksperymenty na własnej sieci.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import RESULTS_DIR


def run_experiments(task_name: str, experiment_grid: dict[str, list[Any]], n_repeats: int = 3) -> None:
    """
    TODO: Osoba 3
    Minimalnie:
    - wczytaj dane
    - zbuduj model
    - odpal wiele konfiguracji
    - powtórz każdy eksperyment kilka razy
    - zapisz wyniki do CSV
    """
    raise NotImplementedError("To implementuje Osoba 3.")


def get_default_results_path(task_name: str) -> Path:
    output_dir = RESULTS_DIR / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{task_name}_results.csv"
