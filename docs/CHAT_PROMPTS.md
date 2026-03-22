# Prompty dla kolejnych osób z zespołu

## Prompt dla Osoby 2 — implementacja sieci

Pracujemy nad projektem SSN na studia. Sieć ma być napisana własnoręcznie w Python + NumPy, bez Keras/PyTorch do uczenia. Dane i preprocessing są już gotowe.

Kontekst techniczny:
- projekt ma dwa taski: `classification_divorce` i `regression_children`
- gotowe dane są generowane przez `src/data_loader.py`
- pliki `.npz` zawierają `X_train`, `X_test`, `y_train`, `y_test`
- klasyfikacja binarna ma target 0/1
- regresja ma target w kształcie `(n,1)`

Chcę, żebyś pomógł mi napisać plik `src/neural_network.py` i ewentualnie `src/layers.py` oraz `src/losses.py`, ale bez przesadnego rozbijania na dziesiątki plików.
Potrzebuję:
1. forward pass
2. backward pass
3. obsługa kilku funkcji aktywacji
4. inicjalizacja wag
5. loss dla regresji i klasyfikacji
6. `fit()` i `predict()`
7. możliwie prosty, czytelny kod pod projekt studencki
8. zgodność z danymi przygotowanymi przez `src/data_loader.py`

Nie twórz wielkiej architektury enterprise. Kod ma być prosty, czytelny i gotowy do eksperymentów.

## Prompt dla Osoby 3 — eksperymenty

Pracujemy nad projektem SSN. Własna sieć w NumPy już istnieje albo zaraz powstanie. Dane przygotowuje `src/data_loader.py`.
Potrzebuję pomocy w napisaniu:
- `src/experiments.py`
- ewentualnie `src/metrics.py`
- zapisu wyników do CSV/JSON

Założenia:
- taski: `classification_divorce` i `regression_children`
- eksperymenty mają badać wpływ parametrów zgodnie z wytycznymi
- dla każdego parametru minimum 4 wartości
- każdy eksperyment ma być powtórzony kilka razy z różnymi seedami
- wyniki muszą zawierać train i test
- kod ma być prosty i łatwy do pokazania na studiach

Chcę, żebyś zaproponował sensowny schemat eksperymentów, strukturę wyników i gotowy kod.

## Prompt dla Osoby 4 — porównanie i raport

Pracujemy nad projektem SSN. Mamy własną sieć w NumPy i preprocessing danych w `src/data_loader.py`.
Potrzebuję:
1. porównania z gotową biblioteką (`sklearn.neural_network.MLPClassifier` / `MLPRegressor`)
2. pomocy w napisaniu raportu
3. tabel i wniosków

Ważne:
- gotowa biblioteka ma być tylko benchmarkiem
- raport ma uczciwie opisać, że dataset może mieć słaby sygnał
- trzeba uwzględnić wymagania projektu: regresja, klasyfikacja, literatura, analiza parametrów, train/test, powtórzenia eksperymentów

Przygotuj prosty, akademicki styl i praktyczny plan raportu.

## Prompt uniwersalny — kontynuacja projektu

To jest projekt studencki z SSN. Używamy repozytorium ze strukturą:
- `src/config.py`
- `src/data_loader.py`
- `data/processed/*.npz`
- `docs/TEAM_HANDOFF.md`

Zakładaj ten interfejs jako prawdę bazową.
Nie proponuj przebudowy wszystkiego od zera.
Buduj rozwiązanie tak, żeby pasowało do istniejącej struktury i mogło zostać łatwo zintegrowane przez 4-osobowy zespół.
