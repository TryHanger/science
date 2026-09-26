# TASKS

## Active

### Пакет v4: T-004 + T-005 + T-002 → Kaggle-прогон v4
Решения: ADR-006 (строки на английском), ADR-007 (resume + scheduler + датасет артефактов), ADR-008 (абляция concat).
- [x] ТЗ-A+B — английские строки: `src/config.py`, `src/data.py`, `src/model.py`, `src/extract_features.py`, `src/predict.py`, `scripts/download_sample.py`, `tests/test_smoke.py` (REVIEW ✓: менялись только docstring/строки, pytest 6/6)
- [x] ТЗ-C — `src/train.py`: `--resume`, `--scheduler`, `--fusion`, `--epochs`, `--patience`, именование артефактов; `cfg.paths.output_dir` (REVIEW ✓)
- [x] ТЗ-D — `src/evaluate.py`: английские строки + оценка всех найденных моделей (mul/concat/q-only), кривые по всем историям, `predictions_concat.csv`, колонка Best Epoch (REVIEW ✓; подписи display_examples исправлены вручную)
- [x] (REVIEW ✓) ТЗ-E — `notebooks/kaggle_run.ipynb` v4 (восстановление артефактов, resume до 40 эпох, concat 15 эпох, исправленный путь в predict-ячейке) + `kernel-metadata.json` (+ датасет артефактов) + `configs/*.yaml` (ключи scheduler)
- [x] ТЗ-F — `README.md` (рус., UTF-8) + `.gitattributes` + `.editorconfig` (REVIEW ✓; хвостовые пробелы убраны вручную)
- [x] ТЗ-H — приватный датасет `tryhanger1/vqa-science-artifacts` из `outputs/kaggle/outputs/` (REVIEW ✓: ready, 7 файлов, размеры совпадают, в публичном поиске отсутствует)
- [x] RETEST — pytest 6/6; resume со старого чекпоинта; evaluate по 3 моделям; веса v3 загружаются в текущие модели без расхождений ключей; `git diff --check` чист
- [x] COMMIT — `805f272`, `5c73917`, `a5da89e` (локально, ahead 3)
- [ ] push в GitHub (подтверждение пользователя) → `kaggle kernels push` v4 → разбор результатов

## Backlog
- T-006 — Живая загрузка в PostgreSQL: задать `PGUSER`/`PGPASSWORD`, выполнить `scripts/load_predictions.py` + `sql/analysis.sql` для `kaggle_v3` (и v4 после прогона).

## Completed
- T-003 (2026-09-26) — `sql/schema.sql`, `sql/analysis.sql`, `scripts/load_predictions.py`; dry-run на `kaggle_v3` ✓ (20 000 строк; is_correct-accuracy 42.12%).
- T-001 (2026-09-26) — Полный прогон на Kaggle GPU (kernel v3). Приёмка ✓: VQA (mul) 43.37% > Question-Only 39.09%. Артефакты: `outputs/kaggle/outputs/`.
- T-000 — Базовый каркас VQA-проекта (src, конфиги local/kaggle, smoke-тесты, ноутбук, локальный прогон).
