# Visual Question Answering (VQA v2) Baseline

Воспроизводимый и масштабируемый baseline по визуальному вопросно-ответному поиску (Visual Question Answering) на базе PyTorch, оптимизированный под структуру научной статьи формата **IMRAD** (*Introduction, Methods, Results, and Discussion*).

Проект готов к запуску как локально (на CPU/GPU с мини-выборкой для быстрой отладки), так и на облачных GPU-инстансах **Kaggle** (Tesla T4 / P100) для проведения полноценных серий экспериментов на данных VQA v2 и MS COCO 2014.

---

## 1. Архитектура baseline-модели

Модель решает задачу классификации ответа на предварительно заданный словарь Top-K наиболее частых ответов датасета VQA v2 по паре `(изображение, текстовый вопрос)`.

```
[ Изображение ] ---> ResNet-50 (avgpool) ---> [2048] ---> Linear(1024) + Tanh ---+
                                                                               |
                                                                        [ Слияние / Fusion ]
                                                                      (Hadamard 'mul' / Concat)
                                                                               |
                                                                        Dropout(0.5)
                                                                               |
                                                                          Linear(1024)
                                                                               |
                                                                             ReLU
                                                                               |
                                                                        Dropout(0.5)
                                                                               |
[ Текст вопроса ] -> Embedding(300) -> LSTM(512) -> [512] -> Linear(1024) + Tanh -+-> Linear(num_answers) -> Logits
                       (с pack_padded_sequence)
```

### Основные компоненты:
1. **Визуальный энкодер (Image Encoder):** Предобученная свёрточная нейросеть `ResNet-50` (веса ImageNet), у которой удалён финальный классификационный полносвязный слой `fc`. Слой `avgpool` возвращает вектор размерности 2048, который проецируется в общее мультимодальное пространство размерности 1024 с нелинейной активацией `Tanh`.
2. **Текстовый энкодер (Question Encoder):** Токенизированный вопрос (длиной до 14 токенов) обрабатывается обучаемым слоем `Embedding(vocab_size, 300)` и рекуррентной сетью `LSTM(hidden_size=512)`.
   * **Защита от влияния `<pad>` токенов:** Используется `pack_padded_sequence(..., enforce_sorted=False)` по реальной длине вопроса `length`. Скрытое состояние $h_T$ извлекается строго из последнего содержательного слова, предотвращая искажение градиентов нулевыми токенами заполнения.
   * Полученный вектор проецируется в пространство размерности 1024 с активацией `Tanh`.
3. **Слияние модальностей (Multimodal Fusion):**
   - По умолчанию (`mul`): поэлементное произведение Адамара (*element-wise multiplication* $v \odot q$).
   - Альтернатива (`concat`): конкатенация векторов $[v; q]$ с последующей линейной проекцией.
4. **Классификатор:** Полносвязный двухслойный MLP с `Dropout(0.5)` и функцией активации `ReLU`, предсказывающий логиты распределения по фиксированному словарю ответов (`Linear(num_answers)`).
5. **Контрольный baseline (Question-Only Baseline):** Абляционная модель, лишённая визуального входа (содержит только текстовый энкодер с `pack_padded_sequence` и классификатор). Служит обязательным контролем для разделов *Results* и *Discussion*, позволяя измерить влияние статистического языкового смещения датасета (*language bias*) и подтвердить вклад визуальных признаков.

---

## 2. Методика для IMRAD: данные и метрики

### Предобработка ответов (Answer Preprocessing)
В пайплайне реализована стандартизированная нормализация ответов согласно официальному протоколу VQA Evaluation:
- Приведение числительных, записанных словами, к цифрам (`"two"` $\to$ `"2"`).
- Удаление артиклей английского языка (`a`, `an`, `the`).
- Удаление пунктуации и приведение текста к нижнему регистру с сохранением сокращений.
- Построение целевых меток (`soft targets`): по 10 аннотациям людей рассчитывается целевая вероятность $s = \min(\frac{n}{3}, 1)$, где $n$ — количество аннотаторов, давших данный нормализованный ответ. Модель обучается с функцией потерь `BCEWithLogitsLoss`.

### Метрика качества (Evaluation Metric)
В разделе **Methods** научной статьи методика фиксируется следующим определением:
> *«We evaluate models using simplified VQA accuracy: $\min(\text{# humans}/3, 1)$, averaged across questions»*.

*(Пояснение: полная версия официальной метрики VQA усредняет совпадения по 10 комбинациям из 9 аннотаторов. Упрощённая форма $\min(n/3, 1)$ даёт практически идентичные результаты, существенно ускоряя вычисления при валидации).*

---

## 3. Структура проекта

```
vqa-project/
├── configs/
│   ├── local.yaml              # Параметры локального запуска (300 train / 100 val, batch 32, 2 эпохи)
│   └── kaggle.yaml             # Параметры Kaggle GPU (80k train / 20k val, batch 256, 15 эпох)
├── src/
│   ├── config.py               # Загрузка YAML-конфигураций, сиды, автоопределение устройства (CUDA/CPU)
│   ├── data.py                 # Токенизация, словари, нормализация ответов, VQADataset, DataLoaders
│   ├── extract_features.py     # Извлечение признаков ResNet-50 в HDF5 (.h5) с поддержкой resume
│   ├── model.py                # PyTorch-модели: VQAModel (mul/concat) и QuestionOnlyBaselineModel
│   ├── train.py                # Обучение с BCEWithLogitsLoss, Early Stopping, --resume и --scheduler
│   ├── evaluate.py             # Оценка моделей (mul/concat/q-only), learning curves, predictions.csv
│   └── predict.py              # Инференс по произвольному изображению и вопросу
├── scripts/
│   ├── download_sample.py      # Загрузка сэмпла изображений и вопросов для локальной отладки
│   └── load_predictions.py     # Валидация CSV и загрузка предсказаний в PostgreSQL (--dry-run)
├── sql/
│   ├── schema.sql              # DDL-схема таблицы vqa_predictions (run_tag, model_name, question_id)
│   └── analysis.sql            # Аналитические SQL-запросы для оценки результатов и калибровки
├── notebooks/
│   └── kaggle_run.ipynb        # Воспроизводимый ноутбук v4 для запуска полного пайплайна на Kaggle
├── tests/
│   └── test_smoke.py           # Смоук-тесты: токенизация, pack_padded_sequence, forward/backward
├── kernel-metadata.json        # Метаданные Kaggle Kernel (tryhanger1/vqa-science)
├── CLAUDE.md                   # Протокол оркестратора (единый источник истины для Claude Desktop)
├── AGENTS.md                   # Контракт и операционные ограничения исполнителя (Antigravity)
├── TASKS.md                    # Реестр задач проекта (Active / Backlog / Completed)
├── IMPLEMENTATION.md           # Фактическое состояние проекта, окружение и результаты запусков
├── DECISIONS.md                # Архитектурные решения проекта (Architecture Decision Records, ADR)
├── requirements.txt            # Зависимости Python
├── .editorconfig               # Стандарты форматирования файлов (UTF-8, LF, отступы)
├── .gitattributes              # Настройки Git для LF-окончаний строк и бинарных форматов
└── .gitignore                  # Исключение data/, outputs/, кэша и артефактов из Git
```

---

## 4. Локальный запуск (Local Debug)

### 4.1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4.2. Загрузка демонстрационного сэмпла данных
Скрипт скачивает 300 обучающих и 100 валидационных вопросов с аннотациями VQA v2 и соответствующие изображения MS COCO 2014:
```bash
python scripts/download_sample.py
```

### 4.3. Запуск смоук-тестов
```bash
pytest -v tests/test_smoke.py
```

### 4.4. Извлечение признаков изображений (ResNet-50 $\to$ HDF5)
Извлечение признаков выполняется один раз с сохранением в `outputs/*.h5`. При повторном запуске скрипт автоматически продолжает работу с места остановки:
```bash
python src/extract_features.py --env local
```

### 4.5. Обучение моделей
1. Обучение мультимодальной модели VQA со слиянием Адамара (`mul` по умолчанию):
```bash
python src/train.py --env local --model-type vqa
```
2. Обучение контрольной модели Question-Only:
```bash
python src/train.py --env local --model-type question_only
```
3. Обучение VQA со слиянием через конкатенацию (`concat`):
```bash
python src/train.py --env local --model-type vqa --fusion concat
```
4. Дообучение из сохранённого чекпоинта с шедулером learning rate:
```bash
python src/train.py --env local --model-type vqa --resume --epochs 5 --scheduler plateau --patience 3
```

### 4.6. Оценка качества и формирование артефактов
Скрипт автоматически находит все сохранённые чекпоинты (`mul`, `concat`, `question_only`) и рассчитывает метрики:
```bash
python src/evaluate.py --env local
```
Скрипт генерирует следующие файлы в каталоге `outputs/`:
- `outputs/metrics_table.csv` — сводная таблица VQA Accuracy по типам вопросов (*overall*, *yes/no*, *number*, *other*) с колонкой `Best Epoch`.
- `outputs/learning_curves.png` — графики функций потерь и точности по эпохам для всех моделей.
- `outputs/predictions.csv` — детальные предсказания валидационного набора для модели VQA `mul`.
- `outputs/predictions_concat.csv` — предсказания для модели VQA `concat`.
- Терминальный вывод первых 10 верных и 10 ошибочных примеров предсказаний.

### 4.7. Инференс на одиночном примере
Запуск через CLI:
```bash
python src/predict.py --env local --question "what color is the object?"
```
Либо с явным указанием пути к изображению:
```bash
python src/predict.py --env local --image data/val2014/COCO_val2014_000000000001.jpg --question "what color is the object?"
```
Вызов функции инференса из Python:
```python
from src.predict import predict

ans, conf = predict("data/val2014/COCO_val2014_000000000001.jpg", "is the object red?", env="local")
print(f"Ответ: {ans}, уверенность: {conf:.2%}")
```

---

## 5. Запуск на Kaggle GPU

### 5.1. Управление через Kaggle CLI
Запуск и мониторинг производятся из корневого каталога репозитория:
```bash
# 1. Отправка и запуск ноутбука на Kaggle
kaggle kernels push -p .

# 2. Проверка статуса выполнения
kaggle kernels status tryhanger1/vqa-science

# 3. Скачивание артефактов и логов по завершении
kaggle kernels output tryhanger1/vqa-science -p outputs/kaggle --force
```

### 5.2. Синхронизация кода
Ноутбук `notebooks/kaggle_run.ipynb` при выполнении клонирует исходный код репозитория с GitHub (`origin/main`).

**Перед запуском `kaggle kernels push` все изменения должны быть закоммичены и отправлены в удалённый репозиторий (`git push`).**

### 5.3. Подключённые датасеты (`kernel-metadata.json`)
В метаданных ядра настроены три источника данных:
1. `biminhco/vqa-v2-question` — вопросы и аннотации train/val VQA v2.
2. `jeffaudi/coco-2014-dataset-for-yolov3` — архив изображений MS COCO 2014 (`train2014`, `val2014`).
3. `tryhanger1/vqa-science-artifacts` — приватный датасет артефактов предыдущих прогонов (`train_img_features.h5`, `val_img_features.h5`, `vocab.json`, чекпоинты `best_model.pth` и `best_question_only_model.pth`, истории метрик). Ноутбук копирует их в `/kaggle/working/outputs/`, позволяя пропускать ресурсоёмкое извлечение признаков и выполнять дообучение через `--resume`.

### 5.4. Сохраняемые артефакты в `/kaggle/working/outputs/`
По завершении прогона в выходной директории сохраняются:
- `train_img_features.h5`, `val_img_features.h5` — предвычисленные признаки ResNet-50.
- `vocab.json` — словари слов вопросов и классов ответов.
- `best_model.pth` — чекпоинт лучшей эпохи VQA baseline (`mul`).
- `best_model_concat.pth` — чекпоинт лучшей эпохи VQA (`concat`).
- `best_question_only_model.pth` — чекпоинт лучшей эпохи Question-Only baseline.
- `metrics_history_vqa.json`, `metrics_history_vqa_concat.json`, `metrics_history_question_only.json` — детальные логи обучения по эпохам (train loss, val loss, val accuracy, learning rate).
- `metrics_table.csv` — таблица финальных метрик по категориям ответов.
- `predictions.csv`, `predictions_concat.csv` — предсказания валидационного набора.
- `learning_curves.png` — итоговые графики кривых обучения.

---

## 6. База данных PostgreSQL (`predictions.csv`)

### 6.1. Схема данных `predictions.csv`
Файл `outputs/predictions.csv` формируется модулем `evaluate.py` и содержит следующие поля:
- `question_id` (BIGINT): уникальный идентификатор вопроса VQA v2.
- `image_id` (BIGINT): идентификатор изображения MS COCO 2014.
- `question` (TEXT): текст вопроса на английском языке.
- `predicted_answer` (VARCHAR): предсказанный нормализованный ответ.
- `confidence` (REAL): уверенность модели (вероятность класса после Softmax в диапазоне $[0.0, 1.0]$).
- `is_correct` (BOOLEAN): флаг корректности ответа (VQA accuracy $\ge 0.5$).
- `answer_type` (VARCHAR): тип вопроса (`yes/no`, `number`, `other`).

### 6.2. DDL-схема таблицы (`sql/schema.sql`)
Таблица `vqa_predictions` расширена полями идентификации эксперимента и модели:
```sql
CREATE TABLE IF NOT EXISTS vqa_predictions (
    run_tag VARCHAR(50) NOT NULL DEFAULT 'local',
    model_name VARCHAR(50) NOT NULL DEFAULT 'vqa_mul',
    question_id BIGINT NOT NULL,
    image_id BIGINT NOT NULL,
    question TEXT NOT NULL,
    predicted_answer VARCHAR(100) NOT NULL,
    confidence REAL NOT NULL,
    is_correct BOOLEAN NOT NULL,
    answer_type VARCHAR(20) NOT NULL,
    PRIMARY KEY (run_tag, model_name, question_id)
);

CREATE INDEX IF NOT EXISTS idx_vqa_predictions_run_tag_answer_type
ON vqa_predictions (run_tag, answer_type);
```

### 6.3. Валидация и загрузка предсказаний (`scripts/load_predictions.py`)
Скрипт проверяет целостность данных (отсутствие пропусков, валидность диапазонов, уникальность первичных ключей) и выполняет пакетную загрузку в PostgreSQL. Подключение конфигурируется стандартными переменными окружения `libpq` (`PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`).

Валидация файла без записи в базу данных (`--dry-run`):
```bash
python scripts/load_predictions.py --csv outputs/predictions.csv --run-tag local --model-name vqa_mul --dry-run
```

Загрузка результатов прогона в PostgreSQL:
```bash
python scripts/load_predictions.py --csv outputs/predictions.csv --run-tag kaggle_v3 --model-name vqa_mul
```

### 6.4. Аналитические SQL-запросы (`sql/analysis.sql`)
Запуск сформированного аналитического скрипта через `psql` с передачей тега эксперимента `run_tag`:
```bash
psql -v run_tag='kaggle_v3' -f sql/analysis.sql
```

Скрипт вычисляет ключевые метрики для разделов *Results* и *Discussion*:
1. Точность и количество ответов по моделям и типам вопросов (`yes/no`, `number`, `other`).
2. Калибровка модели: распределение уверенности по диапазонам (0.0–0.2, 0.2–0.4, ..., 0.8–1.0) с вычислением средней уверенности и реальной точности.
3. Топ-20 наиболее часто предсказываемых ответов и их точность.
4. Топ-20 наиболее уверенных ошибок модели (`is_correct = false`, упорядоченных по убыванию `confidence`).
5. Точность ответов в зависимости от первого слова вопроса (выявление языковых корреляций).
