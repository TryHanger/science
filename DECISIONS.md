# DECISIONS (ADR)

## ADR-001 — Предвычисление признаков ResNet-50 в HDF5
**Статус:** принято (ретроспективно). **Контекст:** ограничение времени GPU на Kaggle. **Решение:** замороженный ResNet-50, признаки avgpool один раз пишутся в `.h5` с resume. **Последствия:** обучение быстрое, но fine-tuning энкодера и attention по регионам недоступны.

## ADR-002 — Fusion по умолчанию: Hadamard (`mul`), `concat` — альтернатива
**Статус:** принято (ретроспективно). **Обоснование:** классический baseline (Antol et al. / Lu et al.), меньше параметров, чем concat.

## ADR-003 — Упрощённая метрика VQA: min(#humans/3, 1)
**Статус:** принято. **Обоснование:** достаточно для baseline; отличие от официального 10-choose-9 оговаривается в Methods.

## ADR-004 — Question-Only абляция как обязательная часть Results
**Статус:** принято. **Обоснование:** контроль language bias.

## ADR-005 — Деплой на Kaggle через Kaggle CLI (`kernels push`) вместо ручного запуска ноутбука
**Статус:** принято (T-001). **Обоснование:** воспроизводимость и автоматизация по протоколу CLAUDE.md. Код подтягивается `git clone` внутри ноутбука → в metadata нужен `enable_internet: true`. `kernel-metadata.json` лежит в корне репо (`tryhanger1/vqa-science`, private, GPU). Датасеты: `biminhco/vqa-v2-question` (вопросы+аннотации), `jeffaudi/coco-2014-dataset-for-yolov3` (изображения); пути резолвятся динамически через `resolve_kaggle_paths` (rglob по `/kaggle/input`), поэтому хардкод в `configs/kaggle.yaml` — только fallback. **Риск:** в Kaggle уходит код из `origin/main` — перед push обязательно синхронизировать GitHub.

## ADR-006 — Строки в коде на английском, документация на русском в UTF-8
**Статус:** принято (T-004), 2026-09-26. **Контекст:** вся кириллица в репозитории потеряна ещё в первом коммите `eda502b` (в файлах буквально `?`), видно на графиках Kaggle. Восстановить исходный текст нельзя. **Решение:** комментарии, docstring, логи, сообщения об ошибках и подписи графиков в `src/`, `scripts/`, `tests/`, `notebooks/` — на английском (ASCII-safe, не зависит от кодировки консоли Windows/Kaggle). `README.md` переписывается на русском в UTF-8 без BOM. Добавить `.gitattributes` (`* text=auto eol=lf`, `*.py/*.md/*.ipynb/*.yaml working-tree-encoding` не задавать) и `.editorconfig` (`charset = utf-8`). **Последствия:** логи и графики читаются везде; тексты статьи IMRAD пишутся отдельно.

## ADR-007 — Дообучение через resume чекпоинтов v3 и перенос артефактов приватным Kaggle-датасетом
**Статус:** принято (T-005), 2026-09-26. **Контекст:** в v3 VQA (mul) ещё растёт на 15-й эпохе (43.37%), early stopping не сработал; yes/no 65.7% при сбалансированных предсказаниях — это не баг (у Question-Only на VQA v2 в литературе ~67% yes/no), а недообучение. **Решение:**
- `train.py` получает `--resume`: грузит `model_state_dict` + `optimizer_state_dict` + `epoch` + `val_acc` (+ `scheduler_state_dict`, если есть) из чекпоинта этой модели, обрезает сохранённую историю до `epoch` чекпоинта и продолжает с `epoch+1` до `--epochs` (общее число эпох).
- LR-scheduler `ReduceLROnPlateau(mode="max", factor=0.5, patience=2)` по val_acc; включается `training.lr_scheduler: plateau` / `--scheduler plateau`, по умолчанию `none` (обратная совместимость).
- v4: VQA(mul) и Question-Only дообучаются одинаково (resume, до 40 эпох, plateau, early stopping patience 5) — абляция остаётся честной.
- Артефакты v3 (`*.h5`, `vocab.json`, `best_model.pth`, `best_question_only_model.pth`, `metrics_history_*.json`) выгружаются в **приватный** датасет `tryhanger1/vqa-science-artifacts`, подключаются в `dataset_sources`; первая ячейка ноутбука копирует их в `/kaggle/working/outputs`. Извлечение признаков пропускается (resume в `extract_features.py`). Словарь строится детерминированно из тех же первых 80k вопросов, `vocab.json` копируется для надёжности.
**Альтернативы:** обучение с нуля на 40 эпохах (дороже по GPU, теряем v3); `kernel_sources` на тот же kernel (самоссылка — поведение Kaggle не гарантировано). **Ограничение:** выбор лучшей эпохи и отчёт — по одному val-сплиту (20k); для baseline допустимо, оговорить в Methods.

## ADR-009 — Разделение ролей Claude Desktop / Antigravity и места хранения правил
**Статус:** принято, 2026-09-26 (по указанию пользователя). **Контекст:** правила были размазаны: общий протокол в `CLAUDE.md` (без REVIEW/TEST/COMMIT, с нерелевантными Qiskit/PennyLane и шагом `kaggle kernels init`), операционные правила agy — в `TASKS.md` и `IMPLEMENTATION.md`, контракт исполнителя повторялся в каждом ТЗ. Итог — параллельные вызовы agy, которые не стартовали, и лишний расход контекста. **Решение:** workflow ANALYZE→PLAN→IMPLEMENT→TEST→REVIEW→FIX→RETEST→DOCUMENT→COMMIT с владельцами этапов; `CLAUDE.md` — только протокол оркестратора; `AGENTS.md` — контракт исполнителя (agy загружает его сам, в ТЗ не дублируется); `TASKS.md` — только реестр; `IMPLEMENTATION.md` — только факты; последовательные вызовы agy по умолчанию; timeout не считается ошибкой. **Альтернативы:** держать всё в `CLAUDE.md` (agy его не читает → дублирование в prompt); `GEMINI.md` вместо `AGENTS.md` (эквивалентно, но `AGENTS.md` — нейтральное имя).

## ADR-008 — Абляция fusion (T-002): concat с нуля по расписанию v3
**Статус:** принято, 2026-09-26. **Решение:** `train.py --fusion concat` переопределяет `model.fusion_method`. Имена артефактов: `mul` → `best_model.pth`, `metrics_history_vqa.json` (как раньше); `concat` → `best_model_concat.pth`, `metrics_history_vqa_concat.json`, `predictions_concat.csv`. Concat обучается с нуля 15 эпох без scheduler — ровно как mul в v3, сравнение mul@15 (из истории v3) vs concat@15. `evaluate.py` оценивает все найденные модели и строит кривые по всем историям.
