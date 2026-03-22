# Handoff dla zespołu

## Co już zrobiła Osoba 1

Osoba 1 przygotowała:
- strukturę projektu,
- loading danych,
- preprocessing,
- zapis gotowych splitów,
- podstawowe EDA.

## Najważniejsza zasada

Nie róbcie własnych, niezgodnych wersji preprocessingu w 4 różnych plikach.
Jeśli ktoś potrzebuje danych:
1. bierze `src/data_loader.py`,
2. wybiera task z `src/config.py`,
3. generuje `.npz`,
4. pracuje na tych samych macierzach co reszta.

## Ustalony interfejs danych

### Wejście
Uruchomienie:
```bash
python -m src.run_data_prep --task classification_divorce
python -m src.run_data_prep --task regression_children
```

### Wyjście
W `data/processed/` powstają pliki:
- `classification_divorce.npz`
- `classification_divorce_metadata.json`
- `classification_divorce_feature_names.txt`
- `regression_children.npz`
- `regression_children_metadata.json`
- `regression_children_feature_names.txt`

### Format `.npz`
Plik `.npz` zawiera:
- `X_train`
- `X_test`
- `y_train`
- `y_test`

### Format targetu
- klasyfikacja binarna: `y_train`, `y_test` to liczby całkowite 0/1
- regresja: `y_train`, `y_test` mają kształt `(n, 1)`

## Kontrakt dla Osoby 2 — implementacja sieci

Twoja sieć powinna umieć przyjąć:
- `X_train: np.ndarray` o kształcie `(n_samples, n_features)`
- `y_train: np.ndarray`
- listę warstw, np. `[input_dim, 16, 8, 1]`
- nazwę funkcji aktywacji
- learning rate
- epochs
- batch size
- typ problemu: `classification` / `regression`

Minimalne metody:
```python
model = NeuralNetwork(...)
history = model.fit(X_train, y_train, X_val=None, y_val=None)
y_pred = model.predict(X_test)
```

Dobrze, jeśli `fit()` zwraca słownik:
```python
{
    "train_loss": [...],
    "val_loss": [...],
}
```

## Kontrakt dla Osoby 3 — eksperymenty

Osoba 3 powinna zapisywać wyniki do CSV/JSON w formacie zbliżonym do:
```text
task,run_id,param_name,param_value,train_metric,test_metric,epochs,learning_rate,batch_size,hidden_layers,hidden_units
```

Dzięki temu Osoba 4 zrobi łatwo raport i wykresy.

## Kontrakt dla Osoby 4 — raport i porównanie

Osoba 4 niech użyje:
- `sklearn.neural_network.MLPClassifier`
- `sklearn.neural_network.MLPRegressor`

To ma być tylko benchmark porównawczy, bo główna sieć ma być własna.

## Twarde wnioski o datasetcie

Na szybkich baseline'ach ten dataset wygląda na słabo przewidywalny:
- `Divorce_Status` jest niezbalansowany
- proste modele są blisko klasy większości / zgadywania
- regresja też wygląda słabo

To trzeba potraktować jako **ryzyko projektowe**, nie jako temat tabu.
