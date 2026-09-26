# IMPLEMENTATION

_Обновлено: 2026-09-26_

## Проект
Baseline для Visual Question Answering (VQA v2 + MS COCO 2014) на PyTorch под статью формата IMRAD. Репозиторий: `github.com/TryHanger/science`.

## Структура
```
configs/    local.yaml (300/100, batch 32, 2 эпохи) · kaggle.yaml (80k/20k, batch 256, 15 эпох, top-1000 ответов)
src/        config · data · extract_features · model · train · evaluate · predict
scripts/    download_sample.py — сэмпл данных для локальной отладки
tests/      test_smoke.py
notebooks/  kaggle_run.ipynb — git clone → extract → train vqa → train question_only → evaluate → predict
data/       локальный сэмпл (30 train / 15 val изображений + JSON VQA v2)
outputs/    h5-признаки, best_model.pth, best_question_only_model.pth, vocab.json, метрики, predictions.csv
```

## Архитектура
ResNet-50 (avgpool, 2048) → Linear(1024)+Tanh ⊙ LSTM(512, pack_padded_sequence) → Linear(1024)+Tanh → MLP (Dropout 0.5) → логиты по top-K ответам. Loss: BCEWithLogits на soft targets. Метрика: min(#humans/3, 1). Абляция: Question-Only.

## Что реально работает
- Локальный пайплайн отработал end-to-end (признаки, обучение обеих моделей, evaluate, CSV).
- Локальные метрики (выборка крошечная, статистически незначимы): VQA mul overall 46.67% vs Question-Only 40.0% (разница — только за счёт `number`).

## Результаты Kaggle (T-001, kernel v3, 80k train / 20k val, 15 эпох)
| Модель | overall | yes/no | number | other |
|---|---|---|---|---|
| VQA Baseline (mul) | **43.37** | 67.15 | 30.43 | 28.97 |
| Question-Only | 39.09 | 66.59 | 29.00 | 21.11 |
- Выигрыш от изображения: +4.28 п.п. overall, почти целиком за счёт `other` (+7.86); на yes/no и number модель опирается в основном на язык.
- Лучшие эпохи: VQA — 15 (ещё растёт), Question-Only — 14 (39.09, плато). Early stopping не сработал ни разу.
- Артефакты локально: `outputs/kaggle/outputs/` (веса 90.6/66.6 MB, h5 98.6/24.8 MB, predictions.csv 1.4 MB, metrics, curves), лог `outputs/kaggle/vqa-science.log` (без ошибок).
- Кириллица в подписях графика — `?` (исходники испорчены с первого коммита; исправляется ADR-006).
- yes/no 65.7% при сбалансированных предсказаниях (3777 yes / 3656 no) — не баг: Question-Only на VQA v2 в литературе ≈ 67%.

## Результаты Kaggle v4 (2026-09-26, 80k/20k; артефакты `outputs/kaggle_v4/outputs/`)
| Модель | overall | yes/no | number | other | Best epoch | Как обучалась |
|---|---|---|---|---|---|---|
| VQA mul | **48.13** | 71.70 | 29.38 | 35.47 | 39 / 40 | v3 (15 эп., LR 1e-3) + resume до 40, plateau (LR 1e-3→5e-4 с эп. 29→2.5e-4 с эп. 39) |
| VQA concat | 46.78 | 68.83 | 28.06 | 35.25 | 15 / 15 | с нуля 15 эп., LR 1e-3, без scheduler |
| Question-Only | 40.57 | 67.96 | 28.93 | 23.09 | 22 / 27 | v3 (14 эп.) + resume, early stop на 27 |
- Разрыв mul − Question-Only: **+7.56 п.п.** overall (v3: +4.28); other +12.38, yes/no +3.74, number +0.45.
- При равных 15 эпохах: concat 46.78 vs mul 43.37 → concat сходится заметно быстрее (+3.4 п.п.); concat на 15-й эпохе ещё растёт, его потолок не измерен.
- mul вышел на плато: после 29-й эпохи +0.25 п.п.; val_loss минимален на 21-й эпохе (0.00392) и дальше растёт при стабильной accuracy — переобучение по уверенности, не по точности.
- Question-Only на плато с ~20-й эпохи (≈40.5%).
- Лог kernel через CLI 2.2.4 скачивается пустым (0 байт); ход обучения подтверждается историями (`lr` = null до resume).

## Статус Kaggle-интеграции
- `kernel-metadata.json` в корне (коммит ffd5966): `tryhanger1/vqa-science`, private, GPU, internet; датасеты `biminhco/vqa-v2-question` (вопросы+аннотации train/val — проверено) и `jeffaudi/coco-2014-dataset-for-yolov3` (`coco2014/images/{train,val}2014`).
- `src/config.py::resolve_kaggle_paths` динамически находит файлы в `/kaggle/input`.
- Kernel `tryhanger1/vqa-science`: v1 упал (нет `kernelspec` в ноутбуке, исправлено локально), v3 — COMPLETE. `kaggle kernels status` отвечает 404, если у kernel нет сессий.
- **Kaggle Output (проверено по коду 2026-09-25):** `output_dir=/kaggle/working/outputs` (абсолютный путь) → попадает в Output версии. Сохраняются: `train_img_features.h5`, `val_img_features.h5`, `vocab.json`, `best_model.pth`, `best_question_only_model.pth` (state_dict + optimizer + epoch + val_acc + cfg), `metrics_history_vqa.json`, `metrics_history_question_only.json`, `metrics_table.csv`, `predictions.csv`, `learning_curves.png`. Логи — только stdout в логе kernel (`kaggle kernels output` → `vqa-science.log`) и в сохранённом `__notebook__.ipynb`/`__results__.html`; отдельных файлов логов скрипты не пишут. Лишнее: клон репо `/kaggle/working/science` (с `.git`) тоже уходит в Output.
- Локально: PostgreSQL 17 (`postgresql-x64-17`, `C:\Program Files\PostgreSQL\17\bin\psql.exe`); env `PGUSER`/`PGPASSWORD` не заданы.

## Пакет v4 (локально готов, 2026-09-26)
- Строки в коде — английские (ADR-006); README — русский UTF-8; `.gitattributes` (LF, бинарные форматы), `.editorconfig`.
- `evaluate.py` оценивает все найденные чекпоинты (mul / concat / q-only), `metrics_table.csv` с колонкой `Best Epoch`, `predictions_concat.csv`, общие кривые.
- Ноутбук v4 (11 ячеек): восстановление артефактов v3 из `/kaggle/input` → extract (пропуск) → resume mul и q-only до 40 эпох (plateau, patience 5) → concat с нуля 15 эпох → evaluate → демо predict (в try/except) → сводка.
- Приватный датасет `tryhanger1/vqa-science-artifacts` (ready): 2 h5, vocab, 2 чекпоинта v3, 2 истории. Подключён в `kernel-metadata.json`. Локальная копия загрузки — `outputs/kaggle_artifacts_upload/` (в .gitignore).
- `origin/main` = `ffd5966` — пакет v4 ещё не в GitHub.

## Возможности train.py (после ТЗ-C)
`--model-type {vqa,question_only} --fusion {mul,concat} --epochs N(общее) --patience N --scheduler {none,plateau} --resume`. Артефакты: mul → `best_model.pth`/`metrics_history_vqa.json`; concat → `best_model_concat.pth`/`metrics_history_vqa_concat.json`; q-only → `best_question_only_model.pth`/`metrics_history_question_only.json`. История пишется после каждой эпохи, включает `lr`. Resume стартует с лучшей эпохи чекпоинта (история обрезается до неё).

## Локальное обучение на GPU (ADR-010)
- GPU: NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM (≈3.2 GB свободно), драйвер 616.92; CUDA работает нативно в Windows (torch 2.5.1+cu124, torchvision 0.20.1+cu124, cuDNN 9.1). WSL (Ubuntu, docker-desktop) установлен, но не используется.
- Конфиг `configs/local_gpu.yaml` (данные/модель = kaggle.yaml, `output_dir ./outputs/local_gpu`, `features.batch_size 64`, `num_workers 2`); все скрипты принимают `--config`.
- `scripts/check_cuda.py` — диагностика (проверено: GPU-тест OK, пик 239 MB).
- `extract_features.py` пропускает извлечение без каталога изображений, если все признаки уже есть в H5.
- Для полноценного прогона не хватает полных JSON VQA v2 в `data/full/` (сейчас в `data/` только сэмпл ~0.5 MB).

## Окружение и инструменты (факты)
- Python 3.11: `C:\Users\Nikita\AppData\Local\Programs\Python\Python311\python.exe`; torch 2.5.1+cu124 (CUDA есть), numpy 2.0.2, pandas 3.0.2, protobuf 7.36.2 (последние два конфликтуют со streamlit/tensorflow — проект не затронут).
- Kaggle CLI 2.2.4 + kagglesdk 0.1.37, аккаунт `tryhanger1`, токен `KGAT_…` (CLI < 1.8 такой токен не принимает → 401).
- `rg` 15.2 установлен (winget).
- Antigravity: `C:\Users\Nikita\AppData\Local\agy\bin\agy.exe` v1.2.11; MCP-мост `C:\Users\Nikita\antigravity-cli-mcp` (bun), env `AGY_PATH`, `AGY_WORKSPACE_ROOT=C:\Users\Nikita\Desktop\sience`, `AGY_TIMEOUT_MS=600000`. Автоматически читает `AGENTS.md` из корня репо.
- Наблюдаемое поведение agy (2026-09-25/26): вызовы с `skip_permissions` блокируются политикой безопасности; без него — работает. Параллельные вызовы не выполняются одновременно (лишние не стартуют). После MCP-таймаута процесс `agy --print` продолжает работу и доводит задачу. Процесс `agy -i` — интерактивная сессия пользователя. Kaggle agy использует через локальный CLI.
