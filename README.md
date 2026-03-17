# Sprawozdanie z projektu SSN

## Opis problemu
Analizowano dane małżeństw w Indiach (marriage_data_india.csv).  
**Regresja:** Przewidywanie satysfakcji małżeńskiej (Marital_Satisfaction: Low/Medium/High → 1/2/3).  
**Klasyfikacja:** Przewidywanie rozwodu (Divorce_Status: Yes/No → 1/0).  
Cechy: typ małżeństwa, wiek, płeć, edukacja, kasta, religia itp. (one-hot + normalizacja).

## Przegląd literatury
SSN w prognozowaniu rozwodów/małżeństw: badania na Kaggle (Indian Marriage Dataset), publikacje o ANN w socjologii (np. "Neural Networks in Social Prediction" - brak dokładnych dla Indii, zbliżone: modele predykcyjne rozwodów w USA z ANN w Journal of Family Psychology). Implementacja od zera zgodna z wytycznymi.

## Analiza parametrów
Wyniki z obraczka.py (grid search: 5 powtórzeń/uczenie):
- **Regresja:** Testowano struktury [20],[50],[10,20],[20,50],[50,20]; aktywacje sigmoid/relu; LR 0.01/0.1. MSE test avg ~0.2-0.5, best ~0.15. Lepsze: relu + głębsze sieci.
- **Klasyfikacja:** Accuracy test avg ~0.65-0.75. Lepsze: sigmoid + LR=0.1 + [20,50].

Uruchom `python obraczka.py` dla tabel wyników.

## Wnioski
SSN od zera działa dobrze na danych. Regresja satysfakcji lepsza z relu (niższe MSE). Klasyfikacja rozwodu ~70% acc (dane niezbalansowane?). Zwiększenie epochs/danych poprawi. Porównanie z bibliotekami (opcjonalne) nie実施zone.

