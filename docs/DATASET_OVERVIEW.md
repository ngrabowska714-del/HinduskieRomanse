# Dataset overview

## Podstawowe informacje

- liczba rekordów: 10000
- liczba kolumn: 18
- brak braków danych: tak
- kolumny liczbowe: ID, Age_at_Marriage, Children_Count, Years_Since_Marriage
- kolumny kategoryczne: Marriage_Type, Gender, Education_Level, Caste_Match, Religion, Parental_Approval, Urban_Rural, Dowry_Exchanged, Marital_Satisfaction, Divorce_Status, Income_Level, Spouse_Working, Inter-Caste, Inter-Religion

## Kolumny

- `ID`
- `Marriage_Type`
- `Age_at_Marriage`
- `Gender`
- `Education_Level`
- `Caste_Match`
- `Religion`
- `Parental_Approval`
- `Urban_Rural`
- `Dowry_Exchanged`
- `Marital_Satisfaction`
- `Divorce_Status`
- `Children_Count`
- `Income_Level`
- `Years_Since_Marriage`
- `Spouse_Working`
- `Inter-Caste`
- `Inter-Religion`

## Rekomendowane targety

### Klasyfikacja
- `Divorce_Status` — prosty problem binarny, ale niezbalansowany
- `Marital_Satisfaction` — opcjonalny problem wieloklasowy

### Regresja
- `Children_Count`
- `Years_Since_Marriage`

## Szybkie obserwacje

- `Divorce_Status`: {'No': 0.8999, 'Yes': 0.1001}
- `Marital_Satisfaction`: {'Medium': 0.5001, 'High': 0.2993, 'Low': 0.2006}
- `Children_Count` ma 6 poziomów: [0, 1, 2, 3, 4, 5]

## Ryzyko metodologiczne

Ten dataset może mieć słaby sygnał predykcyjny.
W praktyce oznacza to, że:
- wyniki sieci mogą nie być wysokie,
- trzeba mocno pilnować porównań z baseline,
- w raporcie warto odróżnić:
  1. poprawność implementacji,
  2. jakość danych,
  3. realną przewidywalność targetu.

## Co zrobić praktycznie

- dla klasyfikacji raportować nie tylko accuracy, ale też `balanced_accuracy`, `precision`, `recall`, `f1`
- dla regresji raportować `MSE`, `RMSE`, `MAE`, `R2`
- wszystkie eksperymenty robić na identycznym preprocessingu
