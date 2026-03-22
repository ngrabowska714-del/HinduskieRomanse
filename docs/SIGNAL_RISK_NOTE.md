# Szybki check sygnału

To nie jest pełna analiza naukowa, tylko wstępny sanity check.

## Wyniki prostych baseline'ów

### Klasyfikacja: `Divorce_Status`
- accuracy: 0.9
- balanced_accuracy: 0.5
- accuracy klasy większości: 0.9

Wniosek: zwykły model praktycznie nie daje przewagi nad klasą większości.

### Regresja: `Children_Count`
- MAE: 1.4963
- R2: -0.0032

Wniosek: prosty model praktycznie nie łapie sygnału.

## Co z tym zrobić

- nie panikować,
- nie mylić słabego datasetu z błędem implementacji,
- w raporcie jasno opisać ograniczenia zbioru,
- porównywać własną sieć nie tylko między sobą, ale też z prostymi baseline'ami.
