# Jak bezpiecznie dołożyć część 2 do repo

To jest instrukcja dla osoby, która wrzuca starter UM do istniejącego projektu.

## Najbezpieczniejsza metoda
1. Zrób kopię obecnego folderu projektu.
2. Wgraj nowe pliki **obok istniejących**, a nie zamiast całego repo.
3. Nie usuwaj niczego z części 1.
4. Przed pierwszym commitem uruchom tylko szybki smoke test.

## Jakie pliki dochodzą
### Nowe pliki w `src/`
- `src/ml_config.py`
- `src/ml_models.py`
- `src/ml_metrics.py`
- `src/ml_experiments.py`

### Nowy plik w root
- `run_um_experiments.py`

### Nowe pliki w `docs/`
- `docs/UM_TEAM_HANDOFF.md`
- `docs/UM_CHAT_PROMPTS.md`
- `docs/UM_INTEGRATION_GUIDE.md`
- `docs/UM_GROUP_MESSAGE.md`
- `docs/UM_PERSON_MESSAGES.md`

### Nowe miejsce na wyniki
- `results/um/`
- `results/um/plots/`

## Czego NIE nadpisywać
- `src/data_loader.py`
- `src/config.py`
- `src/neural_network.py`
- istniejących wyników SSN
- istniejących testów

## Co sprawdzić po wrzuceniu
```bash
python run_um_experiments.py --list
```

Jeśli to działa, to znaczy, że nowa część jest poprawnie osadzona.

Potem można zrobić próbę jednego ownera:
```bash
python run_um_experiments.py --owner person2 --task classification_divorce
```

## Najlepszy workflow git
```bash
git checkout -b feature/um-starter
# wrzuć nowe pliki
# sprawdź listę eksperymentów
# zrób commit
```

Przykład commita:
```bash
git add src/ml_*.py run_um_experiments.py docs/UM_* results/um/.gitkeep
git commit -m "add UM part starter and shared experiment runner"
```

## Jeśli boisz się coś popsuć
Nie kopiuj całego projektu z nowej paczki na ślepo.
Kopiuj tylko nowe pliki i nowe foldery wynikowe.
