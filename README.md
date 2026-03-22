# Projekt SSN — starter dla grupy

To jest uporządkowany starter pod projekt z **Sztucznych Sieci Neuronowych** napisanych własnoręcznie w Pythonie/NumPy.

Zakres projektu zgodny z wytycznymi:
- trzeba rozwiązać **1 problem regresyjny** i **1 problem klasyfikacyjny**,
- trzeba zbadać wpływ parametrów sieci,
- dla każdego parametru trzeba sprawdzić **co najmniej 4 wartości**,
- uczenie trzeba **powtarzać kilkukrotnie** i raportować wyniki dla **train i test**,
- gotowe biblioteki do budowy/uczenia sieci są dozwolone tylko do **porównania**, nie jako główne rozwiązanie.

## Co już jest gotowe

1. Surowe dane w `data/raw/`
2. Kod do:
   - wczytania danych,
   - walidacji kolumn,
   - kodowania kategorii,
   - skalowania cech liczbowych,
   - podziału na train/test,
   - zapisania gotowych macierzy NumPy do `data/processed/`
3. Skrypt EDA i podstawowy opis datasetu
4. Dokumentacja dla zespołu
5. Gotowe prompty dla innych osób w grupie

## Ważna uwaga o datasetcie

Po szybkim sprawdzeniu sygnału ten zbiór wygląda na **bardzo słabo przewidywalny**:
- `Divorce_Status` jest mocno niezbalansowany (około 90% / 10%)
- proste baseline'y dają wyniki bliskie zgadywaniu lub klasy większości

To nie przekreśla projektu, ale trzeba to uczciwie opisać w raporcie:
- jeśli własna sieć nie daje wysokich wyników, to **nie musi być błąd w kodzie**,
- możliwe, że sam zbiór ma mało relacji między cechami i targetem,
- wnioski metodologiczne nadal można zrobić bardzo dobrze.

Jeśli prowadzący nie wymaga trzymania się tego datasetu za wszelką cenę, warto rozważyć plan B. Jeśli zostajecie przy nim, w raporcie trzeba to nazwać wprost.

## Rekomendowane zadania

### Klasyfikacja
- podstawowy target: `Divorce_Status`
- alternatywa: `Marital_Satisfaction`

### Regresja
- podstawowy target: `Children_Count`
- alternatywa: `Years_Since_Marriage`

## Szybki start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python -m src.eda
python -m src.run_data_prep --task classification_divorce
python -m src.run_data_prep --task regression_children
```

## Struktura

```text
projekt_ssn_starter/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── notebooks/
├── references/
├── results/
├── src/
└── tests/
```

## Kontrakt dla reszty zespołu

Osoba 2, 3 i 4 nie powinny zgadywać formatu danych. Mają używać:
- `src/config.py` — definicje tasków
- `src/data_loader.py` — przygotowanie danych
- `data/processed/*.npz` — gotowe macierze
- `data/processed/*_metadata.json` — opis kolumn i mapowań targetu

## Minimalny workflow dla całej grupy

1. **Osoba 1** przygotowuje dane i zapisuje gotowe splity.
2. **Osoba 2** implementuje `NeuralNetwork` w NumPy.
3. **Osoba 3** robi eksperymenty na ustalonym API.
4. **Osoba 4** robi porównanie ze `sklearn` i sprawozdanie.

## Pliki referencyjne

- `references/Wytyczne_do_Projektu_SSN.pdf`
- `references/ESI_Neuron.xlsx`
