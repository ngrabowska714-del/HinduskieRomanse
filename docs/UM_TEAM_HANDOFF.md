# Część 2 projektu — przekazanie pracy zespołowi

## Co dokładnie wymagają wytyczne
Część 2 ma zawierać:
- skrótowy opis **minimum 4 metod uczenia maszynowego**,
- analizę wpływu parametrów tych metod,
- dla **każdego analizowanego parametru co najmniej 4 wartości**,
- analizy wykonane **oddzielnie dla problemu klasyfikacyjnego i regresyjnego**,
- minimalną liczbę analizowanych parametrów równą liczbie osób w grupie, ale nie mniej niż 3.

W tym projekcie spełniamy to tak:
- metody: Decision Tree, Random Forest, kNN, Gradient Boosting,
- problemy: `classification_divorce` i `regression_children`,
- parametry główne do raportu:
  - `max_depth` (Decision Tree)
  - `n_estimators` (Random Forest)
  - `n_neighbors` (kNN)
  - `learning_rate` (Gradient Boosting)

## Zasada nadrzędna
Nie budujemy drugiego projektu obok. Część 2 dokładamy do istniejącego repo i korzystamy z:
- tych samych danych,
- tego samego preprocessingu,
- tych samych splitów train/test,
- tego samego nazewnictwa tasków.

## Kto co robi

### Osoba 1
- utrzymuje infrastrukturę,
- nie daje rozwalić struktury,
- pilnuje spójności ścieżek i wyników,
- pomaga przy merge'ach.

### Osoba 2
Pracuje na metodach:
- Decision Tree
- Random Forest

Uruchamia swoje eksperymenty poleceniem:
```bash
python run_um_experiments.py --owner person2
```

Jeśli potrzebuje tylko jednego tasku:
```bash
python run_um_experiments.py --owner person2 --task classification_divorce
python run_um_experiments.py --owner person2 --task regression_children
```

Oddaje:
- wyniki CSV,
- wykresy,
- krótki opis co wyszło najlepiej.

### Osoba 3
Pracuje na metodach:
- kNN
- Gradient Boosting

Uruchamia swoje eksperymenty poleceniem:
```bash
python run_um_experiments.py --owner person3
```

Oddaje:
- wyniki CSV,
- wykresy,
- krótki opis co wyszło najlepiej.

### Osoba 4
Nie rusza kodu części 2 na starcie, jeśli nie musi.
Ma zebrać:
- krótki opis 4 metod,
- źródła / literaturę,
- tabelę zbiorczą wyników,
- wnioski do raportu,
- zintegrować część 1 i część 2 w jedno sprawozdanie.

## Czego nie robić
- nie zmieniać datasetu,
- nie zmieniać preprocessingu z części 1,
- nie tworzyć własnych skryptów do wczytywania tych samych danych,
- nie przerabiać struktury projektu,
- nie nadpisywać plików innych osób,
- nie zmieniać nazw tasków.

## Gdzie trafiają wyniki
- surowe wyniki eksperymentów: `results/um/*.csv`
- wykresy: `results/um/plots/*.png`

## Co powinno wejść do raportu
Dla każdej metody i problemu:
- parametr badany,
- 4 wartości parametru,
- wyniki train,
- wyniki test,
- krótki komentarz interpretacyjny.
