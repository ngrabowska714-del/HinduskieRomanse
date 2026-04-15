# Osoba 3 — podsumowanie wyników (kNN i Gradient Boosting)

## kNN — classification_divorce (parametr: n_neighbors)

| n_neighbors | test balanced_accuracy | test F1 |
|-------------|------------------------|---------|
| 3           | 0.503                  | 0.054   |
| 5           | 0.503                  | 0.027   |
| 11          | 0.500                  | 0.000   |
| 21          | 0.500                  | 0.000   |

**Najlepszy wynik:** n_neighbors = 3 (test balanced_accuracy = 0.503).  
Małe k (3) dało marginalnie lepszy wynik od pozostałych. Wraz ze wzrostem liczby sąsiadów model coraz bardziej ignoruje klasę mniejszościową i przewiduje wyłącznie klasę dominującą — recall spada do 0. Problem silnej nierównowagi klas (~90% jedna klasa) sprawia, że wszystkie warianty osiągają wyniki bliskie losowemu klasyfikatorowi.

---

## kNN — regression_children (parametr: n_neighbors)

| n_neighbors | test RMSE | test R² |
|-------------|-----------|---------|
| 3           | 1.975     | -0.345  |
| 5           | 1.865     | -0.198  |
| 11          | 1.783     | -0.096  |
| 21          | 1.744     | -0.048  |

**Najlepszy wynik:** n_neighbors = 21 (najniższe RMSE = 1.744, R² najbliższe 0).  
Większe k daje nieco lepsze predykcje — uśrednianie po większej liczbie sąsiadów redukuje szum. Mimo to żaden wariant nie przekracza R² = 0, co oznacza, że każdy model jest gorszy niż prosta predykcja średniej. Zbiór regresyjny ma słabą liniową strukturę.

---

## Gradient Boosting — classification_divorce (parametr: learning_rate)

| learning_rate | test balanced_accuracy | test F1 |
|---------------|------------------------|---------|
| 0.01          | 0.500                  | 0.000   |
| 0.05          | 0.500                  | 0.000   |
| 0.10          | 0.500                  | 0.000   |
| 0.20          | 0.500                  | 0.000   |

**Najlepszy wynik:** brak wyraźnego lidera — wszystkie learning_rate dają identyczne wyniki na zbiorze testowym (balanced_accuracy = 0.5).  
Model nie uczy się klasy mniejszościowej niezależnie od tempa uczenia. Przyczyną jest ta sama silna nierównowaga klas co w kNN.

---

## Gradient Boosting — regression_children (parametr: learning_rate)

| learning_rate | test RMSE | test R² |
|---------------|-----------|---------|
| 0.01          | 1.704     | -0.001  |
| 0.05          | 1.707     | -0.005  |
| 0.10          | 1.712     | -0.011  |
| 0.20          | 1.720     | -0.019  |

**Najlepszy wynik:** learning_rate = 0.01 (RMSE = 1.704, R² ≈ 0).  
Mniejsze tempo uczenia daje nieco lepszą generalizację — wolniejsze dopasowanie zmniejsza ryzyko przeuczenia na słabosygnałowym zbiorze. Różnice między wariantami są jednak minimalne (zakres RMSE ~0.016).
